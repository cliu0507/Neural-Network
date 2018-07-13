#This is the fine tune of resnet
import numpy as np
import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.utils import plot_model
from keras.optimizers import SGD


batch_size = 64
num_classes = 5
epochs = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cliu_resnet50_trained_model.h5'


#Do train/test data loading
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.33)

train_generator = data_generator.flow_from_directory(
		"./data", 
		target_size=(224, 224), 
		shuffle=True,
		color_mode="rgb",
    	class_mode='categorical', 
    	batch_size=batch_size, 
    	subset="training")

validation_generator = data_generator.flow_from_directory(
		"./data", 
		target_size=(224, 224), 
		shuffle=True, 
		color_mode="rgb",
		class_mode='categorical', 
		batch_size=batch_size, 
		subset="validation")



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

