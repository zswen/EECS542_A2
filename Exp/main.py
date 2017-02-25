import os
import sys
import cv2
from random import shuffle
import time
from multiprocessing import Condition, Lock, Process, Manager
import tensorflow as tf

from step import *
from config import *
sys.path.append('../Util')
from debug import loadDataOneThread
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
	shuffle(train_ims)

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
		net.build(train = train_mode)
		net.loss(learning_rate)
	init = tf.global_variables_initializer()
	sess.run(init)
	if net.load(sess, '../Checkpoints', 'Segmentation_%s' % model_name, []):
		print('LOAD SUCESSFULLY')
	else:
		print('[!!!]No Model Found, Train From Scratch')

	#start training
	current_iter = 1
	avg_loss = []
	while current_iter < max_iter:
		t0 = time.clock()
		print('{iter %d}' % (current_iter))
		if current_iter % 10 == 0:
			print('[#]average seg loss is: %f' % np.mean(avg_loss))
			avg_loss = []
		avg_loss.append(step(sess, net, train_image_batches, cv_empty, cv_full, silent_train))
		current_iter += 1
		if current_iter % snapshot_iter == 0:
			net.save(sess, '../Checkpoints', 'Segmentation_%s' % model_name)
		print('[$] iter timing: %d' % (time.clock() - t0))
	return
	

if __name__ == '__main__':
	main()


