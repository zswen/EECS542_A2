import os
import sys
import cv2
from multiprocessing import Condition, Lock, Process, Manager
import tensorflow as tf

from train import *
from config import *
import pdb

#multiprocess objects
manager = Manager()
batch_lock = Lock()
mutex = Lock()
cv_empty = Condition(mutex)
cv_full = Condition(mutex)
train_image_batches = manager.list()

def getIndex(split):
	if split != 'train' and split != 'val':
		print('wrong split name, enter train|val')
	f = open(os.path.join(data_split_root, split + '.txt'))
	names = f.readlines()
	return manager.list([name[0: -1] for name in names])

def main():
	
	#set device
	if len(sys.argv) != 3 or \
		(sys.argv[1] != 'cpu' and sys.argv[1] != 'gpu'):
		print('wrong arguments, <cpu|gpu> <device_idx(integer)> ')
		return
	
	device = sys.argv[1]
	device_idx = int(sys.argv[2])

	#get data split
	valid_ims = getIndex('val')
	train_ims = getIndex('train')

	batch_current = manager.list([0])
	
	#launch data_loader
	processors = []
	for _ in range(num_processor):
		P = Process(target = loadData,
					args = (train_ims,
			 				train_image_batches, 
			 				batch_current,
			 				batch_lock, 
			 				cv_full, cv_empty,
			 				data_loader_capacity,
			 				resize_threshold))
		P.start()
		processors.append(P)

	#create network
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
					  log_device_placement = False))
	net = FCN32VGG()
	with tf.device('/%s: %d' % (device, device_idx)): 
		net.build(debug = True)
		net.loss(learning_rate)
	init = tf.global_variables_initializer()
	sess.run(init)

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


