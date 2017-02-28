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
	total_loss = []
	for idx, subbatch in enumerate(subbatches):
		[loss, seg_loss, softmax, upscore32, score_fr, fc7, fc6, pool4, score_pool4, fuse_pool4, _] = \
			sess.run([net.loss, net.cross_entropy_mean, net.softmax, net.upscore32, net.score_fr, net.fc7, net.fc6, net.pool4, net.score_pool4, net.fuse_pool4, net.train_op], \
			  				  feed_dict = {net.im_input: np.array(subbatch['images']),
			  			   	   			   net.seg_label: np.array(subbatch['labels']),
			  			   	   			   net.apply_grads_flag: int(idx == len(subbatches) - 1)})
		total_loss.append(seg_loss)
	net.done_optimize()
	if not silent:
		print('\t[!]segmentation loss: %f, total loss: %f' % (seg_loss, loss))
	return np.mean(total_loss)

def main():
	sys.path.append('../Util')
	from prepare_data import getSegLabel, getDetectionLabel
	
	device_idx = 0
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
					  log_device_placement = False))
	net = FCN32VGG()
	with tf.device('/cpu: %d' % device_idx): 
		net.build(train = True)
		net.loss(1e-4, 'Mome', None)
	init = tf.global_variables_initializer()
	sess.run(init)
	for idx, f in enumerate(['2007_000032', '2007_000033']):
		test_img = cv2.imread(os.path.join(image_root, f + '.jpg'))
		test_label = getSegLabel([f], color2Idx, segmentation_root)
		[prediction, loss, conv5_3] = sess.run([net.pred_up, net.loss, net.conv5_3], \
						  				   feed_dict = {net.im_input: np.array([test_img]),
						  			   	   				net.seg_label: np.array([test_label]),
						  			   	   				net.apply_grads_flag: idx % 2})
		pdb.set_trace()
	print(loss)
	cv2.imwrite('test.png', prediction[0, ...])
	return

if __name__ == '__main__':
	main()
