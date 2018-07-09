import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
import os




batch_size = 32
num_classes = 5
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cliu_trained_model.h5'



'''
# Comment this because we will use imagedatagenerator rather than read data natively
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = #Write code here
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

'''

input1=Input(shape = (224,224,3))
conv1=Conv2D(32, (3, 3), padding='same',name="Conv1")(input1)
act1=Activation('relu')(conv1)
conv2=Conv2D(32, (3, 3), padding='valid',name="Conv2")(act1)
act2=Activation('relu')(conv2)
maxpool1=MaxPooling2D(pool_size=(2, 2),name="MaxPooling1")(act2)
dropout1=Dropout(0.25)(maxpool1)

conv3=Conv2D(64, (3, 3), padding='same',name="Conv3")(dropout1)
act3=Activation('relu')(conv3)
conv4=Conv2D(64, (3, 3), padding='valid',name="Conv4")(act3)
act4=Activation('relu')(conv4)
maxpool2=MaxPooling2D(pool_size=(2, 2),name="MaxPooling2")(act4)
dropout2=Dropout(0.25)(maxpool2)

fcn1=Flatten(data_format='channels_last')(dropout2)
fcn1=Dense(512,name="Fully_Connect_Layer1")(fcn1)
act5=Activation('relu')(fcn1)
dropout3=Dropout(0.5)(act5)
fcn2=Dense(num_classes,name="Fully_Connect_Layer2")(dropout3)
pred=Activation('softmax',name="Softmax_Classifier_Layer")(fcn2)


model = Model(inputs=input1, outputs=pred)

plot_model(model, to_file='convolutional_neural_network.png')
print(model.summary())


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



'''
#This is one straightforward way 
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        './data/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode='categorical')

'''


#This is another way of doing train/test data loading
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



model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=validation_generator)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


