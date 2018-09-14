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

'''
{'CallToAction': 0, 'Headline': 1, 'HeroImage': 2, 'Logo': 3}
Found 1022 images belonging to 4 classes.
{'CallToAction': 0, 'Headline': 1, 'HeroImage': 2, 'Logo': 3}
'''



cifar10_label_mapping = {
	0:"CallToAction",
	1:"Headline",
	2:"HeroImage",
	3:"Logo"
}




global model
model = load_model('./saved_models/best_4_class_0.8487.hdf5')


def predict(imgpath):
	test_image = cv2.imread('./evaluation/' + imgpath,1)
	a = cv2.resize(test_image, (32, 32))/255.0
	b = np.expand_dims(a,axis=0)
	result=np.squeeze(model.predict(b))
	print(result)
	np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
	for class_name,prob in zip(cifar10_label_mapping.values(),result):
		print(class_name, "{:.10f}".format(prob* 100) + '%')

while True:
	filename = input('Press enter the layer png filename:')
	try:
		if '.png' in filename or '.jpg' in filename:
			predict(filename)
		else:
			predict(filename+".png")
	except Exception:
		print("no such file")
