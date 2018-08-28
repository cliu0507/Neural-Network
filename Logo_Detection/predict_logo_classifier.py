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
	1:"non-logo",
	0:"logo"
}

global model
model = load_model('./saved_models/logo_detection_ResNet_model.026-0.36.hdf5')


def predict(imgpath):
	test_image = cv2.imread('./evaluation/' + imgpath,1)
	
	a = cv2.resize(test_image, (32, 32))/255.0
	b = np.expand_dims(a,axis=0)
	
	
	result=np.squeeze(model.predict(b))
	print(result)
	'''
	np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
	for class_name,prob in zip(cifar10_label_mapping.values(),result):
		print(class_name, "{:.10f}".format(prob* 100) + '%')
	'''
print("Use some logo jpgs to test")
predict('1.png')


print("\n")
print("Use some non logo jpgs to test")
predict('neg1.png')
