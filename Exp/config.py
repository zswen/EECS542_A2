import numpy as np
from collections import OrderedDict

learning_rate = 1e-5
model_name = 'partialReize'

train_mode = True #False for testing
batch_size = 20
snapshot_iter = 100 #write checkpoint for every @ iterations
data_loader_capacity = 10 #cache how many batches
resize_threshold = 100
max_iter = float('inf')
num_processor = 2 #number of processes loading data
silent_train = True #print loss for every step

image_root = '../TrainVal/VOCdevkit/VOC2011/JPEGImages'
annotation_root = '../TrainVal/VOCdevkit/VOC2011/Annotations'
segmentation_root = '../TrainVal/VOCdevkit/VOC2011/SegmentationClass'
data_split_root = '../TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation'

className2Idx = {'aeroplane':1, 
		'bicycle':2, 
		'bird':3, 
		'boat':4, 
		'bottle':5, 
		'bus':6, 
		'car':7, 
		'cat':8, 
		'chair':9, 
		'cow':10, 
		'diningtable':11, 
		'dog':12, 
		'horse':13, 
		'motorbike':14, 
		'person':15, 
		'pottedplant':16, 
		'sheep':17, 
		'sofa':18, 
		'train':19, 
		'tvmonitor':20
		}
idx2ClassName = {v: k for k, v in className2Idx.items()}

color2Idx = {(0, 0, 128): 1,
		 (0, 128, 0): 2,
		 (0, 128, 128): 3,
		 (128, 0, 0): 4,
		 (128, 0, 128): 5,
		 (128, 128, 0): 6,
		 (128, 128, 128): 7,
		 (0, 0, 64): 8,
		 (0, 0, 192): 9,
		 (0, 128, 64): 10,
		 (0, 128, 192): 11,
		 (128, 0, 64): 12,
		 (128, 0, 192): 13,
		 (128, 128, 64): 14,
		 (128, 128, 192): 15,
		 (0, 64, 0): 16,
		 (0, 64, 128): 17,
		 (0, 192, 0): 18,
		 (0, 192, 128): 19,
		 (128, 64, 0): 20,
		 (192, 224, 224): 21
		}

idx2Color = {v: k for k, v in color2Idx.items()}
