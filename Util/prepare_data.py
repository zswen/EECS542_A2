import os
import cv2
import xml.etree.ElementTree as et
import numpy as np

def getSegLabel(image_names, color2class):
	labels = []
	dir_name = '../TrainVal/VOCdevkit/VOC2011/SegmentationClass'
	for idx, name in enumerate(image_names):
		path = os.path.join(dir_name, name)
		img = cv2.imread(path)
		label = np.zeros(img.shape[0: 2])
		for r in range(img.shape[0]):
			for c in range(img.shape[1]):
				if tuple(img[r, c, :]) != (0, 0, 0):
					label[r, c] = color2class[tuple(img[r, c, :])]
		labels.append(label)
	return np.array(labels)

#0~20 classes, 0 for background
def getDetectionLabel(filename, dict, off_value = -1):
	vec = off_value * np.ones(len(dict) + 1) # add one for background
	tree = et.parse(filename)
	root = tree.getroot()
	for subcatagory in root:
		obj = subcatagory.find('name')
		if obj != None:
			vec[dict[obj.text]] = 1
	return vec

def main():
	getSegLabel(['2007_000032.png'], {})
	return

if __name__ == '__main__':
	main()