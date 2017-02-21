import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('../Exp/config.py')

def visualize(image_mask, if_seg = True):
	if if_seg:
		return
	else:
		plt.imshow(image_mask)
	return