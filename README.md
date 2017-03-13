#EECS542_A2
This is an implementation of Fully Connected Convolutional Neural Network and its application in image segmentation task.
##Requirement
Require Tensorflow 1.0 and some Python Packages, all can be installed with pip

The code should be run with Python 3
##Quick Start
clone the repository and `cd ./Exp`
change the config.py to set different config values
Change Image directory accordingly, by default they should in `ROOT/TrainVal`

To run with GPU: `python main.py gpu 0`

To run with CPU: `python main.py cpu 0`
##Code Layout
All running functions are in Exp folder

The network structure is defined under Network folder

Data preparation codes are in Util folder

Submission folder includes a result analyze script
