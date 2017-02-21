import tensorflow as tf
import sys
import os
import numpy as np
import cv2
from random import shuffle

import time

sys.path.append('../Util')
from prepare_data import getSegLabel, getDetectionLabel, getSubbatch

sys.path.append('../Network')
from fcn import FCN32VGG

from config import *
import pdb

#image_names: ['2007_000032'], not include suffices
#data_loader: [{'images': [], 'label_masks': []}]
def loadData(all_image_names, data_loader, batch_current, \
			 batch_lock, cv_full, cv_empty, \
			 data_loader_capacity = 10, resize_threshold = 20):
	while True:
		images = []
		batch_lock.acquire()
		image_names = all_image_names[batch_current[0] * batch_size: (batch_current[0] + 1) * batch_size]
		batch_current[0] = batch_current[0] + 1
		batch_lock.release()
		label_masks = getSegLabel(image_names, color2Idx, segmentation_root)
		for name in image_names:
			image = cv2.imread(os.path.join(image_root, name + '.jpg'))
			images.append(image)

		#protect shared data
		cv_full.acquire()
		while len(data_loader) >= data_loader_capacity:
			cv_full.wait()
		data_loader.append(getSubbatch(images, label_masks, resize_threshold))
		cv_empty.notify()
		cv_full.release()

		batch_lock.acquire()
		if batch_size * batch_current[0] >= len(all_image_names):
			batch_current[0] = 0
			shuffle(all_image_names)
		batch_lock.release()

#image_inputs: list of images [[w, h, c], ...]
#image_labels: list of images [[w, h, c], ...]
def step(sess, net, data_loader, cv_empty, cv_full, silent = True):
	
	#access shared data
	cv_empty.acquire()
	while len(data_loader) == 0:
		cv_empty.wait()
	subbatches = data_loader.pop(0)
	cv_full.notify()
	cv_empty.release()

	t0 = time.clock()
	for idx, subbatch in enumerate(subbatches):
		[loss, _] = sess.run([net.loss, net.train_op], \
			  				  feed_dict = {net.im_input: np.array(subbatch['images']),
			  			   	   			   net.seg_label: np.array(subbatch['labels']),
			  			   	   			   net.apply_grads_flag: int(idx == len(subbatches) - 1)})
	net.done_optimize()
	print('batch time: %d' % (time.clock() - t0), )
	if not silent:
		print('segmentation loss:', loss)
	return

def main():
	sys.path.append('../Util')
	from prepare_data import getSegLabel, getDetectionLabel
	
	device_idx = 0
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
					  log_device_placement = False))
	net = FCN32VGG()
	with tf.device('/cpu: %d' % device_idx): 
		net.build(debug = True)
		net.loss(1e-4)
	init = tf.global_variables_initializer()
	sess.run(init)
	for idx, f in enumerate(['2007_000032', '2007_000033']):
		test_img = cv2.imread(os.path.join(image_root, f + '.jpg'))
		test_label = getSegLabel([f], color2Idx, segmentation_root)
		[prediction, loss, _] = sess.run([net.pred_up, net.loss, net.train_op], \
						  				   feed_dict = {im_input: np.array([test_img]),
						  			   	   				seg_label: np.array([test_label]),
						  			   	   				apply_grads_flag: idx % 2})
	print(loss)
	cv2.imwrite('test.png', prediction[0, ...])
	return

if __name__ == '__main__':
	main()
