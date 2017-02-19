import tensorflow as tf
import sys
import os
sys.path.append('../Network')

import numpy as np
from fcn import FCN32VGG
import cv2
from config import *
import pdb

def step(sess, net, image_inputs, image_outputs):
	return

def main():
	sys.path.append('../Util')
	from prepare_data import getSegLabel, getDetectionLabel
	
	device_idx = 0
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
					  log_device_placement = False))
	net = FCN32VGG()
	with tf.device('/cpu: %d' % device_idx): 
		im_input = tf.placeholder(tf.float32)
		seg_label = tf.placeholder(tf.int32)
		net.build(im_input, debug = True)
		net.loss(seg_label, 1e-4)
	init = tf.global_variables_initializer()
	sess.run(init)
	for f in ['2007_000032', '2007_000033']:
		test_img = cv2.imread(os.path.join(image_root, f + '.jpg'))
		test_label = getSegLabel([f], color2Idx, segmentation_root)
		[prediction, loss, gradients] = sess.run([net.pred_up, net.loss, net.gradients], \
						  				   feed_dict = {im_input: np.array([test_img]),
						  			   	   				seg_label: np.array([test_label])})
		net.gradients_pool.append(net.gradients)
	net.optimize()
	sess.run([net.train_opt])
	pdb.set_trace()
	print loss
	cv2.imwrite('test.png', prediction[0, ...])
	return

if __name__ == '__main__':
	main()