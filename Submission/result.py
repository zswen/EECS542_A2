import numpy as np
import cv2
import sys
import os
import operator
import pickle
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
			predict_dir = './Result', gt_dir = segmentation_root):
	while current_img[0] < len(img_list):
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

def getWrongPrediction(result_dict):
	wrong = {}
	wrong_prediction = {}
	for key,value in result_dict.items():
		if key[0] != key[1]:
			if not key[0] in wrong:
				wrong[key[0]] = {}
			wrong[key[0]][key[1]] = value

	for key,value in wrong.items():
		tp_dict = {}
		for sub_key, sub_value in value.items():
			if sub_key == 0:
				tp_dict['background'] = sub_value
			elif sub_key == 21:
				tp_dict['uncertain'] = sub_value
			else:
				tp_dict[idx2ClassName[sub_key]] = sub_value
		if key == 0:
			wrong_prediction['background'] = tp_dict
		elif key == 21:
			wrong_prediction['uncertain'] = tp_dict
		else:
			wrong_prediction[idx2ClassName[key]] = tp_dict

	print(wrong_prediction)


			


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

	acc_by_class = defaultdict(int)
	IU_by_class = defaultdict(int)
	for key,value in correct_pix.items():
		tmp_acc = value / float(total_pix[key])
		tmp_IU = value / float(total_pix[key] + wrong_pix[key])
		if key == 0:
			acc_by_class['background'] = tmp_acc
			IU_by_class['background'] = tmp_IU
		else:
			acc_by_class[idx2ClassName[key]] = tmp_acc
			IU_by_class[idx2ClassName[key]] = tmp_IU
		mean_acc += tmp_acc
		mean_IU += tmp_IU
		freq_weighted_IU += total_pix[key]*correct_pix[key]/float(total_pix[key] + wrong_pix[key])
	
	pix_acc = sum(correct_pix.values()) / float(sum(total_pix.values()))
	mean_acc = mean_acc / len(correct_pix)
	mean_IU = mean_IU / len(correct_pix)
	freq_weighted_IU = freq_weighted_IU / sum(total_pix.values())

	sorted_acc_by_class = sorted(acc_by_class.items(), key=operator.itemgetter(1))
	sorted_IU_by_class = sorted(IU_by_class.items(), key=operator.itemgetter(1))
	sorted_acc_by_class.reverse()
	sorted_IU_by_class.reverse()

	print(tabulate([["Pixel Accuracy", pix_acc], 
		["Mean Accuracy", mean_acc], 
		["Mean IU", mean_IU], 
		["Frequency Weighted IU", freq_weighted_IU]]))
	print("Accuracy by Class\n")
	print(tabulate(sorted_acc_by_class, headers = ["Object", "Accuracy"]))
	print("IU by Class\n")
	print(tabulate(sorted_IU_by_class, headers = ["Object", "IU"]))



def main():
	#set targets
	if len(sys.argv) != 3:
		print('wrong arguments, <image_dir> <result_dict_dir> should be under this directory')
		return
	
	image_dir = sys.argv[1]
	result_dict_dir = sys.argv[2]
	if not os.path.exists(result_dict_dir):
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
				 				total_result_dict, image_dir))
			P.start()
			processors.append(P)

		for P in processors:
			P.join()
		handle = open(result_dict_dir, 'wb')
		total_result_dict = dict(total_result_dict)
		pickle.dump(total_result_dict, handle)
		handle.close()
		calcMetric(total_result_dict)
		getWrongPrediction(total_result_dict)
	else:
		handle = open(result_dict_dir, 'rb')
		result_dict = pickle.load(handle)
		handle.close()
		calcMetric(result_dict)
		getWrongPrediction(result_dict)
	return

if __name__ == '__main__':
	main()