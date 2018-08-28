#This is the pascal_voc_parser to parse pascal voc format dataset

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import uuid


def get_data(input_path):
	'''function to read the path to folder storing xml files and get data
	args:
		input_path: string
			absolute path to the folder to store xml files
	returns:
		None
	'''
	all_imgs = [] #list to store all image paths and annotation data
	classes_count = {}
	class_mapping = {}
	data_path = input_path 
	visualise = False  
	annot_path = os.path.join(data_path,"XML_Chang") #annotation folder path
	imgs_path = os.path.join(data_path,"JPG_Chang") #raw image folder path
	#find all annotation xmls from annot_path
	annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]  
	idx = 0
	for annot in annots:
		try:
			idx += 1
			et = ET.parse(annot)
			element = et.getroot()
			element_filename = element.find('filename').text
			element_width = int(element.find('size').find('width').text)
			element_height = int(element.find('size').find('height').text)
			element_objs = element.findall('object')
			if len(element_objs) > 0:
				annotation_data = {'filepath': os.path.join(imgs_path, element_filename),'filename':element_filename, 'width': element_width,
									   'height': element_height, 'bboxes': []}
			for element_obj in element_objs:
				class_name = element_obj.find('name').text
				
				#This is to remove trailing number in class name such as "Headline1","Logo2"
				class_name = class_name.rstrip('1234567890.')
				if class_name not in classes_count:
					classes_count[class_name] = 1
				else:
					classes_count[class_name] += 1
				if class_name not in class_mapping:
					class_mapping[class_name] = len(class_mapping)
				obj_bbox = element_obj.find('bndbox')
				x1 = int(round(float(obj_bbox.find('xmin').text)))
				y1 = int(round(float(obj_bbox.find('ymin').text)))
				x2 = int(round(float(obj_bbox.find('xmax').text)))
				y2 = int(round(float(obj_bbox.find('ymax').text)))
				difficulty = int(element_obj.find('difficult').text) == 1
				annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})       
			all_imgs.append(annotation_data)
			if visualise:
				img = cv2.imread(annotation_data['filepath'])
				for bbox in annotation_data['bboxes']:
					cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
								  'x2'], bbox['y2']), (0, 0, 255))
				cv2.imshow('img', img)
				cv2.waitKey(0)
		except Exception as e:
			print(e)
			continue
	return all_imgs, classes_count, class_mapping


def cut_bndbox_to_img(input_path, output_data_path):
	#Parse XML file folder
	all_imgs, classes_count, class_mapping=get_data(input_path)
	if not os.path.exists(output_data_path):
		os.makedirs(output_data_path)
	for class_name in classes_count.keys():
		if not os.path.exists(os.path.join(output_data_path,str(class_name))):
			os.makedirs(os.path.join(output_data_path,class_name))
	for annotation_data in all_imgs:
		try:
			filepath = annotation_data['filepath']
			filename = annotation_data['filename']
			width = annotation_data['width']
			height = annotation_data['height']
			for bbox in annotation_data['bboxes']:
				png_filepath = os.path.join(output_data_path,bbox['class'],str(bbox['class'])+'_'+str(uuid.uuid4())+'_'+os.path.splitext(filename)[0]+'.png')
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				img = cv2.imread(filepath)
				roi = img[y1:y2,x1:x2]
				cv2.imwrite(png_filepath,roi)
		except Exception as e:
			print(e)
			continue



#This is the test main function
#all_imgs, classes_count, class_mapping= get_data("/Users/cliu/Data")
cut_bndbox_to_img(input_path="/Users/cliu/Data",output_data_path="/Users/cliu/Documents/Github/nn/pascal_voc_util/data")



