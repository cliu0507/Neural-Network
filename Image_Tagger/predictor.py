#predictor class 
import os
import numpy as np
import pickle
import os
import time
import tensorflow as tf
import cv2
import keras
import keras.layers
from keras.models import Model
from keras.models import load_model
np.set_printoptions(precision=10,suppress=True)

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

global model
model = load_model('./saved_models/cifar10_ResNet_model.091-0.33.hdf5')


def predict(imgpath):
	test_image = cv2.imread('./evaluation/' + imgpath,1)
	
	a = cv2.resize(test_image, (32, 32))
	b = np.expand_dims(a,axis=0)
	
	
	result=np.squeeze(model.predict(b))
	
	np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
	for class_name,prob in zip(cifar10_label_mapping.values(),result):
		print(class_name, "{:.10f}".format(prob* 100) + '%')


predict('1.jpg')

