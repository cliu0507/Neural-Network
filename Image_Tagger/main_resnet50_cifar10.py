#This is the fine tune of resnet
import numpy as np
import pickle
import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.utils import to_categorical
from PIL import Image

batch_size = 64
num_classes = 5
epochs = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cliu_resnet50_trained_model_cifar10.h5'


#Do train/test data loading
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict



#Write bitmap to png
def write_png(x,y,label_mapping_dict,folder_path):
	
	#This is the dict to store the name of generated png in each class/subfolder
	pngid_for_label_dict={}

	#Init png id list per class
	for label in label_mapping_dict.values():
		pngid_for_label_dict[label] = 0

	print(pngid_for_label_dict)

	for (imagearray,labelid) in zip(x,y):
		im = Image.fromarray(imagearray)
		if label_mapping_dict[labelid]:
			
			#get to know which class the png belong
			label = label_mapping_dict[labelid]
			
			#subfolder path of each class
			subfolder = os.path.join(folder_path,label)

			if not os.path.exists(subfolder):
				os.makedirs(subfolder)

			#make the imagepath
			imagepath = os.path.join(subfolder,str(pngid_for_label_dict[label])) + '.png'
			im.save(imagepath)

			#increment the name of the png image for next png belonging to such class/label
			pngid_for_label_dict[label] += 1
		else:
			raise ValueError("Can't find matching label")




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


#write train png image to each class subfolder
write_png(x_train,y_train,label_mapping_dict=cifar10_label_mapping,folder_path="./cifar-10-batches-pngs")

'''
#Load the resnet50 exclude last couple fcn layers
base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
#Add a fully connected layer
x = Dense(100, activation='relu')(x)
pred = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=pred)

plot_model(model, to_file='convolutional_neural_network_resnet50.png')

for i, layer in enumerate(model.layers):
   print(i, layer.name)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet layers
for layer in base_model.layers:
	layer.trainable = False

print(model.summary())
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(
		train_generator,
		epochs=epochs,
		validation_data=validation_generator)

# Second, we fine tune top inception blocks, i.e. we will freeze
# the first hundreds of layers and unfreeze the rest:

for layer in model.layers[:164]:
   layer.trainable = False
for layer in model.layers[164:]:
   layer.trainable = True

print(model.summary())
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(
		train_generator,
		epochs=epochs,
		validation_data=validation_generator)

# Save model and weights
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
'''
