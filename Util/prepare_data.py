import os
import cv2
import xml.etree.ElementTree as et
import numpy as np

#return list of label masks, since size may not match
def getSegLabel(image_names, color2class, dir_name):
	labels = []
	for idx, name in enumerate(image_names):
		name = name + '.png'
		path = os.path.join(dir_name, name)
		img = cv2.imread(path)
		label = np.zeros(img.shape[0: 2])
		for r in range(img.shape[0]):
			for c in range(img.shape[1]):
				if tuple(img[r, c, :]) != (0, 0, 0):
					label[r, c] = color2class[tuple(img[r, c, :])]
		labels.append(label)
	return labels

#0~20 classes, 0 for background, return a matrix, with an image in a row
def getDetectionLabel(filenames, dict, dir_name, off_value = -1):
	labels = []
	for idx, file in enumerate(filenames):
		file = file + '.xml'
		path = os.path.join(dir_name, file)
		vec = off_value * np.ones(len(dict) + 1) # add one for background
		tree = et.parse(path)
		root = tree.getroot()
		for subcatagory in root:
			obj = subcatagory.find('name')
			if obj != None:
				vec[dict[obj.text]] = 1
		labels.append(vec)
	return np.array(labels)

def testLabel(image_names, annotation_root, image_root, \
			  segmentation_root, className2Idx, color2Idx, idx2ClassName):
	detectionLabels = getDetectionLabel(image_names, className2Idx, annotation_root)
	segmentationLabels = getSegLabel(image_names, color2Idx, segmentation_root)
	assert detectionLabels.shape[0] == len(segmentationLabels)
	for idx in range(detectionLabels.shape[0]):
		print 'image:', image_names[idx]
		from_exist = (detectionLabels[idx, :] + 1).nonzero()[0]
		print 'existence says this image has    ',
		print [idx2ClassName[it] for it in from_exist]
		print segmentationLabels[idx]
		print 'segmentation says this image has ',
		print [idx2ClassName[it] for it in seg]
		print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
	return

def main():
	import sys
	sys.path.append('../Exp')
	from config import *
	image_names = ['2007_000032', '2007_000033', '2007_000039', '2007_000068', '2007_000170']
	testLabel(image_names, annotation_root, image_root, segmentation_root, 
			  className2Idx, color2Idx, idx2ClassName)
	return

if __name__ == '__main__':
	main()