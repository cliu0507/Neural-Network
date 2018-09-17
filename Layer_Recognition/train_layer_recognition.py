from __future__ import print_function
from optparse import OptionParser
from glob import glob
import os
import time
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization,Conv2D, \
	MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D,ZeroPadding2D
from keras.optimizers import SGD,RMSprop,Adagrad,Adam
from keras.callbacks import TensorBoard,ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras_nn import lr_schedule as lr_schedule
from keras.utils import get_file

#Example command:
#python train_layer_recognition.py --path "./data" --network resnet34 --image_augument True  
#python train_layer_recognition.py --path "./data" --network resnet50 --initialization imagenet --image_augument True
#python train_layer_recognition.py --path "./data" --network resnet50 --initialization imagenet --image_augument True --num_epochs 200

timestr = time.strftime("%Y-%m-%d|%H:%M:%S")
parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("--network", dest="network", help="Backbone network to use.", default='resnet34')
parser.add_option("--initialization", dest="initialization", help="Backbone network initialization.", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=100)
parser.add_option("--batch_size", type="int", dest="batch_size", help="Batch Size.", default=50)
parser.add_option("--config_filename", dest="config_filename", help="Training phase metadata.",default="config.pickle")
parser.add_option("--image_augument", dest="image_augument", help="If using image augument",default=False)



(options, args) = parser.parse_args()

print("Start layer recognition training:\n")
if not options.train_path:   # if filename is not given
	parser.error('Error: Path to training data must be specified. Pass --path to command line')

if options.network == 'resnet34':
	from keras_nn import resnet34 as nn
	image_shape=(32,32,3)
	print("Use ResNet 34")
elif options.network == 'resnet50':
	image_shape=(224,224,3)
	from keras_nn import resnet50 as nn
	print("Use ResNet 50")


output_folder = "run_%s" % timestr
output_folder_path = os.path.join(os.getcwd(), output_folder)
if not os.path.isdir(os.path.join(os.getcwd(), output_folder)):
	os.makedirs(output_folder_path)


# Prepare model model saving directory.
model_name = 'model_layer_recognition.{epoch:03d}-{val_loss:.3f}-{val_acc:.4f}.hdf5'
output_model_path = os.path.join(output_folder_path, model_name)
print(output_model_path)

#train/val data set path
trainset_path = os.path.join(options.train_path,'train')
valset_path = os.path.join(options.train_path,'test')


print("\nParse the data set:")
class_list_in_trainset = glob(trainset_path + '/*') # Reads all the folders in which images are present
class_list_in_trainset = [os.path.basename(path) for path in class_list_in_trainset]
class_list_in_trainset = sorted(class_list_in_trainset) # Sorting them
class_dict_in_trainset = dict(zip(class_list_in_trainset, range(len(class_list_in_trainset))))
print("\nIn training set:")
print(class_dict_in_trainset)


class_list_in_valset = glob(valset_path + '/*') # Reads all the folders in which images are present
class_list_in_valset = [os.path.basename(path) for path in class_list_in_valset]
class_list_in_valset = sorted(class_list_in_valset) # Sorting them
class_dict_in_valset = dict(zip(class_list_in_valset, range(len(class_list_in_valset))))
print("\nIn validation set:")
print(class_dict_in_valset)


#Hyper parameter
batch_size = options.batch_size
num_classes = len(class_dict_in_trainset)
num_epochs = options.num_epochs
image_augument = bool(options.image_augument)
initialization = options.initialization
network = options.network

f = open(os.path.join(output_folder_path,"config.txt"),'w')
f.write("batch_size is %s\n" % batch_size)
f.write("num_classes is %s\n" % num_classes)
f.write("num_epochs is %s\n" % num_epochs)
f.write("image_augument is %s\n" % image_augument)
f.write("network is %s\n" % options.network)
f.write("initialization is %s\n" % initialization)
f.close()


#Define network
input_tensor = Input(shape=image_shape)
x = nn.nn_base(input_tensor=input_tensor,trainable=True)

'''
if initialization == 'imagenet' and network == 'resnet50':
	base_model = Model(inputs=input_tensor,outputs=x,name=network)
	weights_path = os.path.join(os.getcwd(),"pretrain_weights","resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5") 
	base_model.load_weights(weights_path)
elif initialization == 'imagenet' and network == 'resnet34':
	raise ValueError("ResNet 34 doesn't have pretrain weights on imagenet")
x = base_model.output
'''

outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal',name ='final_fc')(x)
model = Model(inputs = input_tensor,outputs = outputs)

if initialization == 'imagenet' and network == 'resnet50':
	weights_path = os.path.join(os.getcwd(),"pretrain_weights","resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5") 
	model.load_weights(weights_path,by_name=True)
elif initialization == 'imagenet' and network == 'resnet34':
	raise ValueError("ResNet 34 doesn't have pretrain weights on imagenet")


model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])


tensorboard = TensorBoard(log_dir=output_folder_path)
checkpoint = ModelCheckpoint(filepath=output_model_path,monitor='val_acc',verbose=1,save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)

lr_scheduler = None
#Choose different learning rate scheduler
if num_epochs <= 50:
	lr_scheduler = lr_schedule.lr_scheduler_50
elif num_epochs <=100:
	lr_scheduler = lr_schedule.lr_scheduler_100
else:
	lr_scheduler = lr_schedule.lr_scheduler_200

schedule_lr = LearningRateScheduler(lr_scheduler,verbose=1)
callbacks = [tensorboard,checkpoint,schedule_lr,reduce_lr]



#Data generator
if image_augument: 
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

else:
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
		rotation_range=90,
		# randomly shift images horizontally
		width_shift_range=0.1,
		# randomly shift images vertically
		height_shift_range=0.1,
		# set range for random shear
		shear_range=0.5,
		# set range for random zoom
		zoom_range=0.2,
		# set range for random channel shifts
		channel_shift_range=0.2,
		# set mode for filling points outside the input boundaries
		fill_mode='nearest',
		# value used for fill_mode = "constant"
		cval=0.,
		# randomly flip images
		horizontal_flip=True,
		# randomly flip images
		vertical_flip=True,
		# set rescaling factor (applied before any other transformation)
		rescale=None,
		# set function that will be applied on each input
		preprocessing_function=preprocess_input,
		# image data format, either "channels_first" or "channels_last"
		data_format = "channels_last"
		# fraction of images reserved for validation (strictly between 0 and 1)
		)

train_generator = datagen.flow_from_directory(
		trainset_path,
		target_size=image_shape[:2],
		batch_size=batch_size,
		class_mode='categorical'
		)

validation_generator = datagen.flow_from_directory(
		valset_path,
		target_size=image_shape[:2],
		batch_size=batch_size,
		class_mode='categorical'
		)


#Fit model and start training
model.fit_generator(train_generator,epochs=num_epochs,validation_data=validation_generator,callbacks=callbacks)


#All Done
print("Layer recognition train completed!")
