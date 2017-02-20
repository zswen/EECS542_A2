import os
import sys
import cv2
from threading import Condition, Lock
import threading
import tensorflow as tf

from train import *
from config import *
import pdb

mutex = Lock()
cv_empty = Condition(mutex)
cv_full = Condition(mutex)
train_image_batches = []

def getIndex(split):
	if split != 'train' and split != 'val':
		print('wrong split name, enter train|val')
	f = open(os.path.join(data_split_root, split + '.txt'))
	names = f.readlines()
	return [name[0: -1] for name in names]

def main():
	#create network
	device_idx = 2
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
					  log_device_placement = False))
	net = FCN32VGG()
	with tf.device('/gpu: %d' % device_idx): 
		net.build(debug = True)
		net.loss(1e-4)
	init = tf.global_variables_initializer()
	sess.run(init)

	#get data split
	valid_ims = getIndex('val')
	train_ims = getIndex('train')

	#launch data_loader
	t = threading.Thread(target = loadData,
						 args = (train_ims,
						 		 train_image_batches,
						 		 cv_full, cv_empty,
						 		 data_loader_capacity))
	t.daemon = True
	t.start()

	current_iter = 0
	#start training
	while current_iter < max_iter:
		print('iter %d' % current_iter)
		if current_iter % 1 == 0:
			silent = False
		else:
			silent = True
		step(sess, net, train_image_batches, cv_empty, cv_full, silent)
		current_iter += 1
	return

if __name__ == '__main__':
	main()


