from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf
import pdb

from base import Model

VGG_MEAN = [103.939, 116.779, 123.68]


class FCN32VGG(Model):

    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            logging.info("Load npy file from '%s'.", vgg16_npy_path)
        if not os.path.isfile(vgg16_npy_path):
            logging.error(("File '%s' not found."), vgg16_npy_path)
            sys.exit(1)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-3
        self.gradients_pool = {}
        self.average_grads = []
        self.im_input = tf.placeholder(tf.float32)
        self.seg_label = tf.placeholder(tf.int32)
        self.apply_grads_num = tf.placeholder(tf.float32)
        self.batch_num = tf.placeholder(tf.int32, shape = [1])
        self.varlist = []
        print("npy file loaded")

    def build(self, train, upsample_mode = 32, num_classes = 22, 
            capacity = 10, random_init_fc8 = False, debug = False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        bgr: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        # Convert RGB to BGR

        self.num_classes = num_classes
        self.capacity = capacity
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        with tf.name_scope('Processing'):

            blue, green, red = tf.split(self.im_input, 3, 3) 
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)

        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

        self.fc6 = self._fc_layer(self.pool5, "fc6")

        if train:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self._fc_layer(self.fc6, "fc7")
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",
                                              num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",
                                           num_classes=num_classes,
                                           relu=False)

        self.pred = tf.argmax(self.score_fr, dimension=3)

        if upsample_mode == 32:
            self.upscore32 = self._upscore_layer(self.score_fr, shape=tf.shape(bgr),
                                           num_classes=num_classes,
                                           debug=debug,
                                           name='up', ksize=64, stride=32)
            predicted_score = tf.slice(self.upscore32, [0, 0, 0, 0], \
                [-1, -1, -1, self.num_classes - 1])
        elif upsample_mode == 16:
            self.upscore2 = self._upscore_layer(self.score_fr,
                                                shape=tf.shape(self.pool4),
                                                num_classes=num_classes,
                                                debug=debug, name='upscore2',
                                                ksize=4, stride=2)

            self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                                 num_classes=num_classes)

            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

            self.upscore32 = self._upscore_layer(self.fuse_pool4,
                                                 shape=tf.shape(bgr),
                                                 num_classes=num_classes,
                                                 debug=debug, name='upscore32',
                                                 ksize=32, stride=16)
            predicted_score = tf.slice(self.upscore32, [0, 0, 0, 0], \
                [-1, -1, -1, self.num_classes - 1])
        elif upsample_mode == 8:
            self.upscore2 = self._upscore_layer(self.score_fr,
                                            shape=tf.shape(self.pool4),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore2',
                                            ksize=4, stride=2)
            self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                                 num_classes=num_classes)
            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

            self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                                shape=tf.shape(self.pool3),
                                                num_classes=num_classes,
                                                debug=debug, name='upscore4',
                                                ksize=4, stride=2)
            self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
                                                 num_classes=num_classes)
            self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

            self.upscore32 = self._upscore_layer(self.fuse_pool3,
                                                 shape=tf.shape(bgr),
                                                 num_classes=num_classes,
                                                 debug=debug, name='upscore32',
                                                 ksize=16, stride=8)
            predicted_score = tf.slice(self.upscore32, [0, 0, 0, 0], \
                [-1, -1, -1, self.num_classes - 1])
        
        self.pred_up = tf.argmax(predicted_score, dimension=3)

    def loss(self, lr, optimizer = 'Mome', head = None):
        #labels is a tf.placeholder [batch, height, width]
        with tf.name_scope('loss'):
            epsilon = tf.constant(value=1e-4)
            self.labels = tf.one_hot(self.seg_label, self.num_classes)
            self.labels_ = tf.reshape(self.labels, [-1, self.num_classes])
            self.upscore32_ = tf.reshape(self.upscore32, [-1, self.num_classes])

            if head is not None:
                cross_entropy = -tf.reduce_sum(tf.multiply(self.labels * tf.log(self.softmax),
                                               head), reduction_indices=[1])
            else:
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                    (labels = self.labels_, logits = self.upscore32_)
                self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy,
                                                name='xentropy_mean')
            
            tf.add_to_collection('losses', self.cross_entropy_mean)

            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

            #optimizer of the net
            if optimizer == 'Mome':
                self.opt = tf.train.MomentumOptimizer(lr, 0.9)
            elif optimizer == 'Adam':
                self.opt = tf.train.AdamOptimizer(learning_rate = lr)
            else:
                assert 0

            #calculate gradients
            self.gradients = self.opt.compute_gradients(self.loss)
            enqueue_ops = []
            with tf.device('/cpu: 0'):
                for grad_and_var in self.gradients:
                    self.gradients_pool[grad_and_var[1]] = \
                        tf.FIFOQueue(self.capacity, tf.float32, shapes = grad_and_var[0].get_shape())
                    enqueue = self.gradients_pool[grad_and_var[1]].enqueue(grad_and_var[0])
                    enqueue_ops.append(enqueue)

            self.enqueue = tf.group(*enqueue_ops)

            for var in self.gradients_pool:
                #grad_and_vars_i is (var0, grad0_imgi)
                with tf.device('/cpu: 0'):
                    grads = self.gradients_pool[var].dequeue_many(self.batch_num)
                grad = tf.concat(grads, 0)
                temp_grad_num = self.apply_grads_num
                for _ in range(len(grad.get_shape()) - 1):
                    temp_grad_num = tf.expand_dims(temp_grad_num, -1)
                grad = tf.div(tf.reduce_mean(tf.multiply(grad, temp_grad_num), 0), 
                    tf.reduce_mean(self.apply_grads_num))
                grad_and_var = (grad, var)
                self.average_grads.append(grad_and_var)
            
            self.train_op = self.opt.apply_gradients(self.average_grads, 
                                      global_step = self.global_step)

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            self.varlist.append(filt)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            self.varlist.append(conv_biases)
            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
            self.varlist.append(filt)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            self.varlist.append(conv_biases)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            num_input = in_features
            stddev = 0.005#(2 / (num_input * 20))**0.5
            # Apply convolution
            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = 0.005#(2 / (num_input * 20))**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        return tf.get_variable(name="biases", initializer=init, shape=shape)

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name='biases', shape=shape,
                               initializer=initializer)

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init, shape=shape)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
