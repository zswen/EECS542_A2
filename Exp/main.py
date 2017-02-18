import tensorflow as tf
from config import *
import os
import sys
sys.path.append('../Util')
from prepare_data import generateSegLabel, getDetectionLabel

def main():
	f = '2007_000032.xml'
	fullpath = os.path.join(annotation_root, f)
	log = open(fullpath, 'r')
	getDetectionLabel(fullpath, className2Idx)
	return

if __name__ == '__main__':
	main()


