import numpy as np
import keras
import keras.layers
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization,Conv2D, \
	MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D,ZeroPadding2D

#convolution block for resNet
def conv_block(input_tensor,filters,block,stage,trainable=True):
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
	conv_name_base = 'res_stage' + str(stage) + '_' + block + '_branch' 
	bn_name_base = 'bn_stage' + str(stage) + '_' + block + '_branch' 

	#Feature map shrinks and dimension increases
	if stage!=1 and block == 'a':
		#first conv in one res block
		x = Conv2D(filters1,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_1',trainable=trainable)(input_tensor)
		x = BatchNormalization(axis=3,name = bn_name_base + '_1')(x)
		x = Activation('relu')(x)

		#second conv in one res block
		x = Conv2D(filters2,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_2',trainable=trainable)(x)
		x = BatchNormalization(axis=3,name = bn_name_base + '_2')(x)
		x = Activation('relu')(x)

		#Use (1,1) conv to increase dimension
		shortcut = Conv2D(filters2,(1,1),strides=(2,2),padding='same',kernel_initializer='he_normal',name = conv_name_base+'_identity',trainable=trainable)(input_tensor)
		shortcut = BatchNormalization(axis=3, name=bn_name_base + '_identity')(shortcut)
	

	#Not the block that dimension increases, strides is always (1,1)
	else:
		#first conv in one res block
		x = Conv2D(filters1,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_1',trainable=trainable)(input_tensor)
		x = BatchNormalization(axis=3,name = bn_name_base + '_1')(x)
		x = Activation('relu')(x)

		#second conv in one res block
		x = Conv2D(filters2,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',name = conv_name_base + '_2',trainable=trainable)(x)
		x = BatchNormalization(axis=3,name = bn_name_base + '_2')(x)
		x = Activation('relu')(x)

		#No need to use (1,1) conv to increase dimension
		shortcut = BatchNormalization(axis=3, name=bn_name_base + '_identity')(input_tensor)

	x = keras.layers.add([x,shortcut])
	x = Activation('relu')(x)
	return x


def nn_base(input_tensor=None, input_shape = (None,None,3),trainable = True):
	
	if input_tensor is None:
		input_tensor = Input(shape=input_shape)

	x = Conv2D(16,(3,3),strides=1,padding='same',kernel_initializer='he_normal',name='conv1',trainable=trainable)(input_tensor)
	x = conv_block(input_tensor=x,filters=[16,16],stage=1,block='a',trainable=trainable)
	x = conv_block(input_tensor=x,filters=[16,16],stage=1,block='b',trainable=trainable)
	x = conv_block(input_tensor=x,filters=[16,16],stage=1,block='c',trainable=trainable)
	
	x = conv_block(input_tensor=x,filters=[32,32],stage=2,block='a',trainable=trainable)
	x = conv_block(input_tensor=x,filters=[32,32],stage=2,block='b',trainable=trainable)
	x = conv_block(input_tensor=x,filters=[32,32],stage=2,block='c',trainable=trainable)
	
	x = conv_block(input_tensor=x,filters=[64,64],stage=3,block='a',trainable=trainable)
	x = conv_block(input_tensor=x,filters=[64,64],stage=3,block='b',trainable=trainable)
	x = conv_block(input_tensor=x,filters=[64,64],stage=3,block='c',trainable=trainable)
	x = GlobalAveragePooling2D()(x)
	return x