import os
import cv2
import numpy as np
import pdb

def generateSegLabel(image_names, color2class):
	labels = []
	dir_name = '../TrainVal/VOCdevkit/VOC2011/SegmentationClass'
	for idx, name in enumerate(image_names):
		path = os.path.join(dir_name, name)
		img = cv2.imread(path)
		label = np.zeros(img.shape[0: 2])
		for r in img.shape[0]:
			for c in img.shape[1]:
				if img[r, c, :] is not 
	return

def main():
	generate_seg_label(['2007_000032.png'], {})
	return

if __name__ == '__main__':
	main()