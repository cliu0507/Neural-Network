'''
# Chang's Implementaion of ResNet34 

# Reference:
 [Deep Residual Learning for Image Recognition](
	https://arxiv.org/abs/1512.03385)

'''



import os
import numpy as np
import pickle
import os
import time
import tensorflow as tf
import cv2
import keras
import keras.layers
from PIL import Image
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D,ZeroPadding2D
from keras.utils import plot_model,to_categorical
from keras.optimizers import SGD,RMSprop,Adagrad,Adam
from keras.callbacks import TensorBoard,ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input


#Learning Rate scheduler function
def lr_scheduler(epoch):
	"""Learning Rate Schedule
	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.
	# Arguments
		epoch (int): The number of epochs
	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 150:
		lr = 1e-6
	elif epoch > 100:
		lr = 1e-5
	elif epoch > 50:
		lr = 1e-4
	print('Learning rate: ', lr)
	return lr




#-------------------------------
#Get the classname and id
from glob import glob
class_names = glob("./data/train/*") # Reads all the folders in which images are present
class_names = sorted(class_names) # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))
print(name_id_map)


#start main function
batch_size = 50
num_classes = len(name_id_map)
epochs = 200
image_shape=(224,224,3) #channel last


# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models_fine_tune_imagenet_resnet50')
model_name = 'component_classification_ResNet_model.{epoch:03d}-{val_acc:.4f}.hdf5'

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)




#-------------------------------
#Build ResNet



#use pre-trained imagenet
base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(x)

'''
#current knowledge is that we need to unlock all layers
for layer in base_model.layers[:-2]:
    layer.trainable = False
'''

model = Model(inputs=base_model.input,outputs=outputs)

#Set pretrained resnet to be freezed

'''
for layer in base_model.layers:
    layer.trainable = False
'''
for layer in model.layers:
	print(layer, layer.trainable)

model.summary()
'''

'''

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])


#plot_model(model, to_file='convolutional_neural_network_resnet50.png')

#-------------------------------
#Set callbacks for each epoch
tensorboard = TensorBoard(log_dir='./logs')
checkpoint = ModelCheckpoint(filepath=filepath,
							 monitor='val_acc',
							 verbose=1,
							 save_best_only=True)
callbacks = [tensorboard,checkpoint]



#-------------------------------
#start reading image data
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format = "channels_last",
        preprocessing_function=preprocess_input
        )

train_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical'
        )

print(train_generator.class_indices)

validation_generator = datagen.flow_from_directory(
        'data/test',
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical'
        )


model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks)


print("ResNet Training Done")