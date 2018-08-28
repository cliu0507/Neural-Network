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
	lr = 5e-4
	if epoch > 200:
		lr = 1e-7
	elif epoch > 150:
		lr = 1e-6
	elif epoch > 100:
		lr = 1e-5
	print('Learning rate: ', lr)
	return lr * 10



#-------------------------------
#start main function

batch_size = 50
num_classes = 5
epochs = 300
image_shape=(32,32,3) #channel last

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'logo_detection_ResNet_model.{epoch:03d}-{val_loss:.2f}.hdf5'
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)





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


#plot_model(model, to_file='convolutional_neural_network_resnet50.png')

#-------------------------------
#Set callbacks for each epoch
tensorboard = TensorBoard(log_dir='./logs')
checkpoint = ModelCheckpoint(filepath=filepath,
							 monitor='val_acc',
							 verbose=1,
							 save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
schedule_lr = LearningRateScheduler(lr_scheduler,verbose=1)
callbacks = [tensorboard,checkpoint,reduce_lr,schedule_lr]



#-------------------------------
#start reading image data
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format = "channels_last",
        validation_split=0.3)

train_generator = datagen.flow_from_directory(
        'data',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'training')

print(train_generator.class_indices)

validation_generator = datagen.flow_from_directory(
        'data',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'validation')

print(validation_generator.class_indices)

model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=500,
        callbacks=callbacks)

from glob import glob
class_names = glob("./data/*") # Reads all the folders in which images are present
class_names = sorted(class_names) # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))


print("ResNet Training Done")