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
from visualize import visualize
import pdb

#multiprocess objects
manager = Manager()
batch_lock = Lock()
mutex = Lock()
cv_empty = Condition(mutex)
cv_full = Condition(mutex)
train_image_batches = manager.list()

def getIndex(split, mgr = manager):
	if split != 'train' and split != 'val':
		print('wrong split name, enter train|val')
	f = open(os.path.join(data_split_root, split + '.txt'))
	names = f.readlines()
	return mgr.list([name[0: -1] for name in names])

def main():
	
	#set device
	if len(sys.argv) != 3 or \
		(sys.argv[1] != 'cpu' and sys.argv[1] != 'gpu'):
		print('wrong arguments, <cpu|gpu> <device_idx(integer)> ')
		return
	
	device = sys.argv[1]
	device_idx = int(sys.argv[2])

	#get data split
	if train_mode:
		split = 'train'
	else:
		split = 'val'
	data_ims = getIndex(split)
	shuffle(data_ims)

	batch_current = manager.list([0])
	
	#launch data_loader
	if train_mode:
		processors = []
		for _ in range(num_processor):
			P = Process(target = loadData,
						args = (data_ims,
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
		net.build(train = train_mode, upsample_mode = upsample_mode)
		net.loss(learning_rate, optimizer_mode, bias)
	init = tf.global_variables_initializer()
	sess.run(init)
	#check model status and decide restore strategy
	if model_init == model_save:
		restore_vars = []
	else:
		restore_vars = net.varlist
	if net.load(sess, '../Checkpoints', 'Segmentation_%s' % model_init, restore_vars):
		print('LOAD SUCESSFULLY')
	elif train_mode:
		print('[!!!]No Model Found, Train From Scratch')
	else:
		print('[!!!]No Model Found, Cannot Test')
		return

	#start training
	if train_mode:
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
				net.save(sess, '../Checkpoints', 'Segmentation_%s' % model_save)
			print('[$] iter timing: %d' % (time.clock() - t0))
	#start testing
	else:
		if not os.path.isdir(result_save_path):
			os.makedirs(result_save_path)
			print('[!]Direction for results not found\n[!]Create %s' % result_save_path)
		for idx, img_name in enumerate(data_ims):
			if not os.path.isfile(os.path.join(result_save_path, img_name + '.png')):
				img = cv2.imread(os.path.join(image_root, img_name + '.jpg'))
				[segmentation] = sess.run([net.pred_up], 
					feed_dict = {net.im_input: np.array([img])})
				visualize(segmentation[0, ...], \
					if_seg = True, save_path = os.path.join(result_save_path, img_name + '.png'))
				print('Save segmentation result of     %s, [%d/%d]' \
					% (img_name, idx + 1, len(data_ims)))
			else:
				print('Detected segmentation result of %s, [%d/%d]' \
					% (img_name, idx + 1, len(data_ims)))

	return
	

if __name__ == '__main__':
	main()


