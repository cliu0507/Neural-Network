import numpy as np
import keras
import keras.layers
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization,Conv2D, \
	MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D,ZeroPadding2D


def identity_block(input_tensor, kernel_size, filters, stage, block,trainable=True):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of
			middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	"""
	
	filters1, filters2, filters3 = filters
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1),kernel_initializer='he_normal',name=conv_name_base + '2a',trainable=trainable)(input_tensor)
	x = BatchNormalization(name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,padding='same',kernel_initializer='he_normal',name=conv_name_base + '2b',trainable=trainable)(x)
	x = BatchNormalization(name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1),kernel_initializer='he_normal',name=conv_name_base + '2c',trainable=trainable)(x)
	x = BatchNormalization(name=bn_name_base + '2c')(x)

	x = keras.layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x


def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2, 2),trainable=True):
	"""A block that has a conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of
			middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
		strides: Strides for the first conv layer in the block.
	# Returns
		Output tensor for the block.
	Note that from stage 3,
	the first conv layer at main path is with strides=(2, 2)
	And the shortcut should have strides=(2, 2) as well
	"""
	
	filters1, filters2, filters3 = filters
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), strides=strides,kernel_initializer='he_normal',name=conv_name_base + '2a',trainable=trainable)(input_tensor)
	x = BatchNormalization(name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same',kernel_initializer='he_normal',name=conv_name_base + '2b',trainable=trainable)(x)
	x = BatchNormalization(name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1),kernel_initializer='he_normal',name=conv_name_base + '2c',trainable=trainable)(x)
	x = BatchNormalization(name=bn_name_base + '2c')(x)

	shortcut = Conv2D(filters3, (1, 1), strides=strides,kernel_initializer='he_normal',name=conv_name_base + '1',trainable=trainable)(input_tensor)
	shortcut = BatchNormalization(
		name=bn_name_base + '1')(shortcut)

	x = keras.layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x




def nn_base(input_tensor=None, input_shape = (None,None,3),trainable = True):

	if input_tensor is None:
		input_tensor = Input(shape=input_shape)
	
	x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
	x = Conv2D(64, (7, 7),strides=(2, 2),padding='valid',kernel_initializer='he_normal',name='conv1',trainable=trainable)(x)
	x = BatchNormalization(name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),trainable=trainable)
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',trainable=trainable)
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',trainable=trainable)

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',trainable=trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',trainable=trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',trainable=trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',trainable=trainable)

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',trainable=trainable)

	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',trainable=trainable)
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',trainable=trainable)
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',trainable=trainable)
	x = GlobalAveragePooling2D()(x)
	return x