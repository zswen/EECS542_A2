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



#[loss, _, labels] = sess.run([net.loss, net.train_op, net.labels],feed_dict = {net.im_input: np.array([image_input]),net.seg_label: np.array([image_labels[idx]]),net.apply_grads_flag: int(idx == len(image_inputs) - 1)})
