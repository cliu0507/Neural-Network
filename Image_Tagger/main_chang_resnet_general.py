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

'''
How to compute the total layers of an Official ResNet50: (Assume input size is (224x224,3))

	one bottom layers:
		7x7 Conv2D + 3x3 Maxpooling2D
		output_size is 112x112
	
	Resdual blocks: 
		stage 1:
			[
				1x1,64
				3x3,64
				1x1,256
			] * 3
		stage 2:
			[
				1x1,128
				3x3,128
				1x1,512
			] * 4
		stage 3:
			[
				1x1,256
				3x3,256
				1x1,1024
			] * 6
		stage 4:
			[
				1x1,512
				3x3,512	
				1x1,1024
			] * 3

		3 * (3+4+6+3) = 48
		output_size is 7*7

	one top layer:
		AveragePooling + 1000 fc + softmax
		output_size is (,10)

'''


def resnet_layer_for_cifar10(
	inputs,
	filters,
	block,
	stage):
	'''A residual conv block which has two convs layers per block (usually use it in deeper network like ResNet50+)
	# Arguments
		input_tensor: input tensor
		filters: list of integers, the filters of 3 conv layers at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	
	'''

	filters1,filters2 = filters
	print(filters)
	

	conv_name_base = 'res_stage' + str(stage) + '_' + block + '_branch' 
	bn_name_base = 'bn_stage' + str(stage) + '_' + block + '_branch' 

	#Feature map shrinks and dimension increases
	if stage!=1 and block == 'a':
		#first conv in one res block
		x = Conv2D(filters1,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_1')(inputs)
		x = BatchNormalization(axis=3,name = bn_name_base + '_1')(x)
		x = Activation('relu')(x)

		#second conv in one res block
		x = Conv2D(filters2,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_2')(x)
		x = BatchNormalization(axis=3,name = bn_name_base + '_2')(x)
		x = Activation('relu')(x)

		#Use (1,1) conv to increase dimension
		shortcut = Conv2D(filters2,(1,1),strides=(2,2),padding='same',kernel_initializer='he_normal',name = conv_name_base+'_identity')(inputs)
		shortcut = BatchNormalization(axis=3, name=bn_name_base + '_identity')(shortcut)
	

	#Not the block that dimension increases, strides is always (1,1)
	else:
		#first conv in one res block
		x = Conv2D(filters1,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_1')(inputs)
		x = BatchNormalization(axis=3,name = bn_name_base + '_1')(x)
		x = Activation('relu')(x)

		#second conv in one res block
		x = Conv2D(filters2,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_2')(x)
		x = BatchNormalization(axis=3,name = bn_name_base + '_2')(x)
		x = Activation('relu')(x)

		#No need to use (1,1) conv to increase dimension
		shortcut = BatchNormalization(axis=3, name=bn_name_base + '_identity')(inputs)

	x = keras.layers.add([x,shortcut])
	x = Activation('relu')(x)
	return x


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
	if epoch > 80:
		lr = 1e-6
	elif epoch > 60:
		lr = 1e-5
	elif epoch > 40:
		lr = 1e-4
	print('Learning rate: ', lr)
	return lr



#-------------------------------
#start main function

batch_size = 32
num_classes = 10
epochs = 100
image_shape=(32,32,3) #channel last
data_augmentation = True

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_ResNet_model.{epoch:03d}-{val_loss:.2f}.hdf5'
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)



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

cifar10_test_datafile_list=[
		"./cifar-10-batches-py/test_batch",
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
#Prepare X_train and Y_train

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
print("Prepare cifar training dataset Done!")



#-------------------------------
#Prepare X_test and Y_test

#Read all data files
x_test_list = []
y_test_list = []
for datafile in cifar10_test_datafile_list:
	x_test_list.append(np.array(unpickle(datafile)[b'data']))
	y_test_list.append(np.array(unpickle(datafile)[b'labels']))

#concatenate train data from multiple datafiles
x_test = np.concatenate(x_test_list,axis=0)
y_test = np.concatenate(y_test_list,axis=0)

#change to one-hot variable
y_test_one_hot = to_categorical(y_test)


#change x_train to (NUM_SAMPLE,3,32,32) - channel first
x_test = np.reshape(x_test,newshape=(-1,3,32,32))

#change x_train to (NUM_SAMPLE,32,32,3) - channel last
x_test = np.moveaxis(x_test,1,-1)
print("Prepare cifar test dataset Done!")



#-------------------------------
#Build ResNet

inputs = Input(shape=image_shape)
x = Conv2D(16,(3,3),strides=1,padding='same',kernel_initializer='he_normal',name='conv1')(inputs)
x = resnet_layer_for_cifar10(inputs=x,filters=[16,16],stage=1,block='a')
x = resnet_layer_for_cifar10(inputs=x,filters=[16,16],stage=1,block='b')
x = resnet_layer_for_cifar10(inputs=x,filters=[16,16],stage=1,block='c')

x = resnet_layer_for_cifar10(inputs=x,filters=[32,32],stage=2,block='a')
x = resnet_layer_for_cifar10(inputs=x,filters=[32,32],stage=2,block='b')
x = resnet_layer_for_cifar10(inputs=x,filters=[32,32],stage=2,block='c')

x = resnet_layer_for_cifar10(inputs=x,filters=[64,64],stage=3,block='a')
x = resnet_layer_for_cifar10(inputs=x,filters=[64,64],stage=3,block='b')
x = resnet_layer_for_cifar10(inputs=x,filters=[64,64],stage=3,block='c')

x = AveragePooling2D(pool_size=8)(x)
x = Flatten()(x)
outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(x)
model = Model(input=inputs,outputs=outputs)
model.summary()
optimizer=Adam(lr_scheduler(0))
model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])


plot_model(model, to_file='convolutional_neural_network_resnet50.png')

#-------------------------------
#Set callbacks for each epoch
tensorboard = TensorBoard(log_dir='./logs_test')
checkpoint = ModelCheckpoint(filepath=filepath,
							 monitor='val_acc',
							 verbose=1,
							 save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
schedule_lr = LearningRateScheduler(lr_scheduler,verbose=1)
callbacks = [tensorboard,checkpoint,reduce_lr,schedule_lr]



#-------------------------------
#Use real time data augmentation:
if data_augmentation:
	#If using data augmentation
	print('Using data augumentation:')
	datagen = ImageDataGenerator(
		# set input mean to 0 over the dataset
		featurewise_center=False,
		# set each sample mean to 0
		samplewise_center=False,
		# divide inputs by std of dataset
		featurewise_std_normalization=False,
		# divide each input by its std
		samplewise_std_normalization=False,
		# apply ZCA whitening
		zca_whitening=False,
		# epsilon for ZCA whitening
		zca_epsilon=1e-06,
		# randomly rotate images in the range (deg 0 to 180)
		rotation_range=0,
		# randomly shift images horizontally
		width_shift_range=0.1,
		# randomly shift images vertically
		height_shift_range=0.1,
		# set range for random shear
		shear_range=0.,
		# set range for random zoom
		zoom_range=0.,
		# set range for random channel shifts
		channel_shift_range=0.,
		# set mode for filling points outside the input boundaries
		fill_mode='nearest',
		# value used for fill_mode = "constant"
		cval=0.,
		# randomly flip images
		horizontal_flip=True,
		# randomly flip images
		vertical_flip=False,
		# set rescaling factor (applied before any other transformation)
		rescale=None,
		# set function that will be applied on each input
		preprocessing_function=None,
		# image data format, either "channels_first" or "channels_last"
		data_format=None,
		# fraction of images reserved for validation (strictly between 0 and 1)
		validation_split=0.0)
	
	#Compute quantities required for featurewise normalization
	#(std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)
	
	model.fit_generator(datagen.flow(x_train, y_train_one_hot, batch_size=batch_size),
						validation_data=(x_test, y_test_one_hot),
						epochs=epochs, verbose=1, workers=4,
						callbacks=callbacks)
else:
	#Not using data augmentation
	print('Not using data augmentation:')
	model.fit(x_train, y_train_one_hot,
		batch_size=batch_size,
		epochs=epochs,
		validation_data=(x_test, y_test_one_hot),
		shuffle=True,
		callbacks=callbacks)

print("ResNet Training Done")