import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
import os
import sys
from config import *
from prepare_data import *
sys.path.append('../Exp/')

def visualize(image_mask, if_seg = True, save_path = '../Submission/test.png'):

	if if_seg:
		[width, height] = image_mask.shape
		seg_img = np.zeros((width, height, 3))
		for i in range(width):
			for j in range(height):
				if int(image_mask[i, j]) != 0:
					seg_img[i, j, :] = idx2Color[int(image_mask[i, j])]

		cv2.imwrite(save_path, seg_img)
	else:
		cv2.imwrite(save_path, image_mask)
	return


def main():
	path = '/Users/zswen/Desktop/EECS542/Assignment2_git/TrainVal/VOCdevkit/VOC2011/SegmentationClass'
	seg = getSegLabel(['2007_000032'], color2Idx, path)
	imput = np.asarray(seg[0])
	visualize(imput, True)
	return

if __name__ == '__main__':
	main()
