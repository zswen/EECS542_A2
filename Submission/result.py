import numpy as np
import cv2
import sys
import os
from tabulate import tabulate
from multiprocessing import Lock, Process, Manager
from collections import defaultdict
sys.path.append('../Exp')
from config import *

import pdb

lock_dict = Lock()
lock_list = Lock()
mgr = Manager()

def analyze(gt_mask, pred_mask):
	assert gt_mask.shape == pred_mask.shape
	result_dict = defaultdict(int)
	(r, c) = gt_mask.shape[0: 2]
	for ir in range(r):
		for ic in range(c):
			result_dict[(color2Idx[tuple(gt_mask[ir, ic, :])], \
						 color2Idx[tuple(pred_mask[ir, ic, :])])] += 1
	return result_dict

def summary(img_list, current_img, locks, total_result_dict, 
			gt_dir = segmentation_root, predict_dir = './Result'):
	while current_img[0] < 100:
		locks[0].acquire()
		idx = current_img[0]
		current_img[0] = current_img[0] + 1
		locks[0].release()
		im = img_list[idx]
		print('Summary %s [%d/%d]' % (im, idx + 1, len(img_list)))
		gt_mask = cv2.imread(os.path.join(gt_dir, im + '.png'))
		pred_mask = cv2.imread(os.path.join(predict_dir, im + '.png'))
		result_dict = analyze(gt_mask, pred_mask)
		locks[1].acquire()
		for k in result_dict:
			if total_result_dict.get(k) is not None:
				total_result_dict[k] += result_dict[k]
			else:
				total_result_dict[k] = result_dict[k] #(i,j): integer
		locks[1].release()

def calcMetric(result_dict): 
	correct_pix = defaultdict(int)
	total_pix = defaultdict(int)
	wrong_pix = defaultdict(int)
	for key,value in result_dict.items():
		if key[0] == key[1]:
			correct_pix[key[0]] = value
		else:
			wrong_pix[key[1]] += value
		total_pix[key[0]] += value

	mean_acc = 0
	mean_IU = 0
	freq_weighted_IU = 0
	for key,value in correct_pix.items():
		mean_acc += value / float(total_pix[key])
		mean_IU += value / float(total_pix[key] + wrong_pix[key])
		freq_weighted_IU += total_pix[key]*correct_pix[key]/float(total_pix[key] + wrong_pix[key])
	
	pix_acc = sum(correct_pix.values()) / float(sum(total_pix.values()))
	mean_acc = mean_acc / len(correct_pix)
	mean_IU = mean_IU / len(correct_pix)
	freq_weighted_IU = freq_weighted_IU / sum(total_pix.values())

	teams_list = ["Pixel Accuracy", "Mean Accuracy", "Mean IU", "Frequency Weighted IU"]
	print(tabulate([["Pixel Accuracy", pix_acc], 
		["Mean Accuracy", mean_acc], 
		["Mean IU", mean_IU], 
		["Frequency Weighted IU", freq_weighted_IU]]))


def main():
	f = open(os.path.join(data_split_root, 'val.txt'))
	names = f.readlines()
	val_ims = mgr.list([name[0: -1] for name in names])
	total_result_dict = mgr.dict()
	current_img = mgr.list([0])
	processors = []
	for _ in range(10):
		P = Process(target = summary,
					args = (val_ims, 
			 				current_img,
			 				[lock_list, lock_dict], 
			 				total_result_dict))
		P.start()
		processors.append(P)

	for P in processors:
		P.join()

	calcMetric(total_result_dict)
	return

if __name__ == '__main__':
	main()