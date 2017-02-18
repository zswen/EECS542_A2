import tensorflow as tf
from config import *
import os
import sys
import cv2
sys.path.append('../Util')
from prepare_data import generateSegLabel, getDetectionLabel

def main():
	f = '2007_000032'
	full_path = os.path.join(annotation_root, f + '.xml')
	image_path = os.path.join(image_root, f + '.jpg')
	seg_path = os.path.join(segmentation_root, f + '.png')
	img = cv2.imread(image_path)
	label_mask = cv2.imread(seg_path)
	log = open(full_path, 'r')
	exist = getDetectionLabel(full_path, className2Idx)
	generateSegLabel(image_names, color2class):
	from_exist = (exist + 1).nonzero()[0]
	print 'This image has ',
	print [idx2ClassName[it] for it in from_exist]
	return

if __name__ == '__main__':
	main()


