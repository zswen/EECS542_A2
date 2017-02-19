import os
import cv2
import xml.etree.ElementTree as et
import numpy as np

def generateSegLabel(image_names, color2class, dir_name):
	labels = []
	for idx, name in enumerate(image_names):
		path = os.path.join(dir_name, name)
		img = cv2.imread(path)
		label = np.zeros(img.shape[0: 2])
		for r in img.shape[0]:
			for c in img.shape[1]:
				if img[r, c, :] is not (0, 0, 0):
					label[r, c] = color2class[img[r, c, :]]
		labels.append(label)
	return np.array(labels)

#0~20 classes, 0 for background
def getDetectionLabel(filenames, dir_name, dict, off_value = -1):
	labels = []
	for file in enumerate(filenames):
		path = os.path.join(dir_name, file)
		vec = off_value * np.ones(len(dict) + 1) # add one for background
		tree = et.parse(path)
		root = tree.getroot()
		for subcatagory in root:
			obj = subcatagory.find('name')
			if obj != None:
				vec[dict[obj.text]] = 1
		labels.append(vec)
	return labels

def main():
	className2Idx = OrderedDict(
	   {'aeroplane':1, 
		'bicycle':2, 
		'bird':3, 
		'boat':4, 
		'bottle':5, 
		'bus':6, 
		'car':7, 
		'cat':8, 
		'chair':9, 
		'cow':10, 
		'diningtable':11, 
		'dog':12, 
		'horse':13, 
		'motorbike':14, 
		'person':15, 
		'pottedplant':16, 
		'sheep':17, 
		'sofa':18, 
		'train':19, 
		'tvmonitor':20
		})
	#generate_seg_label(['2007_000032.png'], {})
	dir_name = '../TrainVal/VOCdevkit/VOC2011/SegmentationClass'
	getDetectionLabel(['2007_000032.png'],dir_name,className2Idx)
	return

if __name__ == '__main__':
	main()