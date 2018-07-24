'''
ResNet 50 on Cifar, Train from sketch

Dataset: Cifar10 - 10 classes

Will do hyperparameter search

Result:


'''

import os
import numpy as np
import pickle
import os
import time
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D
from keras.utils import plot_model,to_categorical
from keras.optimizers import SGD,RMSprop,Adagrad
from keras.callbacks import TensorBoard
from PIL import Image
import tensorflow as tf
import cv2

batch_size = 100
num_classes = 10
epochs = 50


#Do train/test data loading
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict



cifar10_datafile_list=[
		"./cifar-10-batches-py/data_batch_1",
		"./cifar-10-batches-py/data_batch_2",
		"./cifar-10-batches-py/data_batch_3",
		"./cifar-10-batches-py/data_batch_4",
		"./cifar-10-batches-py/data_batch_5"
		]

cifar10_label_mapping = {
	0:"airplane",
	1:"automobile",
	2:"bird",
	3:"cat",
	4:"deer",
	5:"dog",
	6:"frog",
	7:"horse",
	8:"ship",
	9:"truck"
}


#-------------------------------
#Prepare X and Y
#Read all data files
x_train_list = []
y_train_list = []
for datafile in cifar10_datafile_list:
	x_train_list.append(np.array(unpickle(datafile)[b'data']))
	y_train_list.append(np.array(unpickle(datafile)[b'labels']))

#concatenate train data from multiple datafiles
x_train= np.concatenate(x_train_list,axis=0)
y_train=np.concatenate(y_train_list,axis=0)

#change to one-hot variable
y_train_one_hot = to_categorical(y_train)



#change x_train to (NUM_SAMPLE,3,32,32) - channel first
x_train = np.reshape(x_train,newshape=(-1,3,32,32))

#change x_train to (NUM_SAMPLE,32,32,3) - channel last
x_train = np.moveaxis(x_train,1,-1)


#Actually we don't need to resize it to 224 * 224 for resnet

'''
print("Resize the image to 224*224")
x_train_resize = []
#Resize to 224
i = 0
for sample in x_train:
	if i % 1000 == 0:
		print(i)
	x_train_resize.append(cv2.resize(sample, (224, 224)))
	i+=1
x_train_resize = np.array(x_train_resize)
'''
x_train_resize = x_train
print("Data Preparation Done!")

#-------------------------------
#Model Training
#Search learning rate
for transfer_learning in [False,True]:
	for model_name in ['ResNet50']:
		for learning_rate in [1e-3,1e-4]:
			for use_dropout in [False]:
				for top_layer_num_fc in [1,0]:
					for optimizer_name in ['RMSprop']:
						#if we want to use pretrained model for transfer learning
						weights='imagenet' if transfer_learning else None
						
						#which base model do we use
						#base_model = globals()[model_name](weights=weights, include_top=False, input_shape=(224, 224, 3))
						base_model = globals()[model_name](weights=weights, include_top=False,input_shape=(32, 32, 3))
						x = base_model.output
						x = AveragePooling2D((7, 7), name='avg_pool')(x)
						x = Flatten()(x)

						#stack fully connected layers
						if top_layer_num_fc == 3:
							x = Dense(4096, activation='relu')(x)
						if top_layer_num_fc == 2:
							x = Dense(2048, activation='relu')(x)
						if top_layer_num_fc == 1:
							x = Dense(1024, activation='relu')(x)

						#if using dropout
						if use_dropout == True:
							x = Dropout(0.5)(x)

						pred = Dense(num_classes, activation='softmax',name='fc10')(x)
						model = Model(inputs=base_model.input, outputs=pred)
						
						#which optimizer to use and its learning rate
						optimizer=globals()[optimizer_name](lr=learning_rate)
						
						#compile model
						model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
						
						#Freeze the pretrained parameter if enabling transfer learning
						if transfer_learning == True:
							for layer in base_model.layers:
								layer.trainable = False

						#run id:
						run_name = str(model_name)+'|'+'finetune_'+str(transfer_learning)+"|"+"lr_"+str(learning_rate)+'|'+"dropout_"+str(use_dropout) +'|'+'num_fc_'+str(top_layer_num_fc) + '|' +'optimizer_' + optimizer_name
						
						print(run_name)
						
						#Set tensorboard log folder for individual run
						logdir='./logs_new/'+str(run_name)
						tensorboard = TensorBoard(log_dir=logdir)
						
						model.fit(
							x=x_train_resize,
							y=y_train_one_hot,
							epochs=epochs,
							verbose=1,
							validation_split=0.3,
							shuffle = True,
							validation_data=None,
							callbacks = [tensorboard])
						

print("Done")
