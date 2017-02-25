import numpy as np
import cv2
from visualize import *

def checkLabels(tensor_labels):
	[_, width, height, _] = tensor_labels.shape
	out = np.zeros((width, height))
	for i in range(width):
		for j in range(height):
			out[i, j] = np.nonzero(tensor_labels[0, i, j, :])[0][0]

	visualize(out, True, '../Util/debug.png')

def loadDataOneThread(all_image_names, data_loader, batch_current, resize_threshold = 20):
		images = []
		image_names = all_image_names[batch_current[0] * batch_size: (batch_current[0] + 1) * batch_size]
		batch_current[0] = batch_current[0] + 1
		label_masks = getSegLabel(image_names, color2Idx, segmentation_root)
		for name in image_names:
			image = cv2.imread(os.path.join(image_root, name + '.jpg'))
			images.append(image)
		data_loader.append(getSubbatch(images, label_masks, resize_threshold))

		if batch_size * batch_current[0] >= len(all_image_names):
			batch_current[0] = 0
			shuffle(all_image_names)
