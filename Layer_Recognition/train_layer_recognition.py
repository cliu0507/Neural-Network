from __future__ import print_function
from optparse import OptionParser
from glob import glob
import os
import time

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization,Conv2D, \
	MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D,ZeroPadding2D
from keras.optimizers import SGD,RMSprop,Adagrad,Adam
from keras.callbacks import TensorBoard,ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

#Example command:
#python train_layer_recognition.py --path "./data" --network resnet34
#python train_layer_recognition.py --path "./data" --network resnet50

timestr = time.strftime("%Y-%m-%d|%H:%M:%S")
parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("--network", dest="network", help="Backbone network to use.", default='resnet34')
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=100)
parser.add_option("--batch_size", type="int", dest="batch_size", help="Batch Size.", default=50)
parser.add_option("--config_filename", dest="config_filename", help="Training phase metadata.",default="config.pickle")
(options, args) = parser.parse_args()

print("Start layer recognition training:\n")
if not options.train_path:   # if filename is not given
	parser.error('Error: Path to training data must be specified. Pass --path to command line')

if options.network == 'resnet34':
	from keras_nn import resnet34 as nn
	image_shape=(32,32,3)
	print("Use ResNet 34, train from Scratch")
elif options.network == 'resnet50':
	image_shape=(224,224,3)
	from keras_nn import resnet50 as nn
	print("Use ResNet 50, train from Scratch")


output_folder = "run_%s" % timestr
output_folder_path = os.path.join(os.getcwd(), output_folder)
if not os.path.isdir(os.path.join(os.getcwd(), output_folder)):
	os.makedirs(output_folder_path)


# Prepare model model saving directory.
model_name = 'model_layer_recognition.{epoch:03d}-{val_loss:.3f}-{val_acc:.4f}.hdf5'
output_model_path = os.path.join(output_folder_path, model_name)


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
epochs = options.num_epochs


#Define network
input_tensor = Input(shape=image_shape)
x = nn.nn_base(input_tensor=input_tensor,trainable=True)
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(x)
model = Model(inputs = input_tensor,outputs = outputs)
model.summary()


model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
tensorboard = TensorBoard(log_dir=output_folder_path)
checkpoint = ModelCheckpoint(filepath=output_model_path,monitor='val_acc',verbose=1,save_best_only=True)
callbacks = [tensorboard,checkpoint]


#Data generator and data augument
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
model.fit_generator(train_generator,epochs=epochs,validation_data=validation_generator,callbacks=callbacks)


#All Done
print("Layer recognition train completed!")
