import os
import xml.etree.ElementTree as et


def getDetectionLabel(filename,dict):
	vec = [-1]*21
	tree = et.parse(filename)
	root = tree.getroot()
	for subcatagory in root:
		obj = subcatagory.find('name')
		if obj != None:
			vec[dict[obj.text]] = 1
	print vec

def main():
	dict = {'aeroplane':1,'bicycle':2,'bird':3,'boat':4,'bottle':5,'bus':6,'car':7,
				'cat':8,'chair':9,'cow':10,'diningtable':11,'dog':12,'horse':13,'motorbike':14,
					'person':15,'pottedplant':16,'sheep':17,'sofa':18,'train':19,'tvmonitor':20}
	source = '../TrainVal/VOCdevkit/VOC2011/Annotations'
	for root, dirs, filenames in os.walk(source):
		for f in filenames:
			fullpath = os.path.join(source, f)
			log = open(fullpath, 'r')
			getDetectionLabel(fullpath,dict)
			

if __name__ == '__main__':
	main()