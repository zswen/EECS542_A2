import os
import cv2
import xml.etree.ElementTree as et
import numpy as np
import sys
sys.path.append('../Exp')
from config import *
import pdb

from cluster import HierarchicalClustering

def getSubbatch(images, image_labels, similar_thred = 20):
	sizes = [(image.shape[0], image.shape[1], idx) for idx, image in enumerate(images)]
	cl = HierarchicalClustering(sizes, lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]))
	clusters = cl.getlevel(similar_thred)
	subbatches = []
	for cluster in clusters:
		if len(cluster) > 1:
			ideal_size = np.median(cluster, axis = 0)
			ideal_size = [int(i) for i in ideal_size]
			subbatch_im = []
			subbatch_label = []
			for img in cluster:
				if img[0] != ideal_size[0] or img[1] != ideal_size[1]:
					subbatch_im.append(cv2.resize(images[img[2]], (ideal_size[1], ideal_size[0])))
					subbatch_label.append(cv2.resize(image_labels[img[2]], (ideal_size[1], ideal_size[0])))
				else:
					subbatch_im.append(images[img[2]])
					subbatch_label.append(image_labels[img[2]])
			subbatches.append({'images': np.array(subbatch_im), 
							   'labels': np.array(subbatch_label)})
		else:
			subbatches.append({'images': np.array([images[cluster[0][2]]]), 
							   'labels': np.array([image_labels[cluster[0][2]]])})
	print([subbatch['images'].shape[0] for subbatch in subbatches])
	return subbatches


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
		print('image:', image_names[idx])
		from_exist = (detectionLabels[idx, :] + 1).nonzero()[0]
		print('existence says this image has    ',)
		print([idx2ClassName[it] for it in from_exist])
		seg = np.unique(segmentationLabels[idx])[1: -1]
		print('segmentation says this image has ',)
		print([idx2ClassName[int(it)] for it in seg])
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	return

def main():
	image_names = ['2007_000032', '2007_000033', '2007_000039', '2007_000068', '2007_000170']
	testLabel(image_names, annotation_root, image_root, segmentation_root, 
			  className2Idx, color2Idx, idx2ClassName)
	return

if __name__ == '__main__':
	main()
