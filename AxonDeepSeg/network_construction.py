# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Get rid of Tensorflow warnings.
import tensorflow as tf
import pickle
from .data_management.input_data import input_data
from .config_tools import generate_config
import AxonDeepSeg.ads_utils


# Definition of functions


# ------------------------ LAYERS

def conv_relu(x, n_out_chan, k_size, k_stride, scope, 
              w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              training_phase=True, activate_bn = True, bn_decay = 0.999, keep_prob=1.0):
    '''
    Default data format is NHWC.
    Initializers for weights and bias are already defined (default).
    :param training_phase: Whether we are in the training phase (True) or testing phase (False)
    '''
    
    with tf.variable_scope(scope):
        if activate_bn == True:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=k_size, stride=k_stride, 
                                       activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale':True, 'is_training':training_phase,
                                                          'decay':bn_decay,'scope':'bn'},
                                       weights_initializer = w_initializer, scope='convolution'
                                      )
        else:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=k_size, stride=k_stride, 
                                       activation_fn=tf.nn.relu, weights_initializer = w_initializer, scope='convolution'
                                      )
        net = tf.contrib.layers.dropout(net, keep_prob=keep_prob, is_training=training_phase)
        tf.add_to_collection('activations',net)
        return net
    
def downconv(x, n_out_chan, scope, 
              w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              training_phase=True, activate_bn = True, bn_decay = 0.999):
    
    '''
    Default data format is NHWC.
    Initializers for weights and bias are already defined (default).
    :param training_phase: Whether we are in the training phase (True) or testing phase (False)
    '''
    
    with tf.variable_scope(scope):
        if activate_bn == True:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=5, stride=2, 
                                       activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale':True, 'is_training':training_phase,
                                                          'decay':bn_decay, 'scope':'bn'},
                                       weights_initializer = w_initializer, scope='convolution'
                                      )
        else:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=5, stride=2, 
                                       activation_fn=tf.nn.relu, weights_initializer = w_initializer, scope='convolution'
                                      )
        
        tf.add_to_collection('activations',net)
        return net
    
def upconv(x, n_out_chan, scope, 
              w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              training_phase=True, activate_bn = True, bn_decay = 0.999):
    
    '''
    Default data format is NHWC.
    Initializers for weights and bias are already defined (default).
    :param training_phase: Whether we are in the training phase (True) or testing phase (False)
    '''
    
    with tf.variable_scope(scope):
        if activate_bn == True:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=3, stride=1, 
                                       activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale':True, 'is_training':training_phase,
                                                          'decay':bn_decay, 'scope':'bn'},
                                       weights_initializer = w_initializer, scope='convolution'
                                      )
        else:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=3, stride=1, 
                                       activation_fn=tf.nn.relu, weights_initializer = w_initializer, scope='convolution'
                                      )
        
        tf.add_to_collection('activations',net)
        return net
    
    

def maxpool(x, k_size, k_stride, scope, padding='VALID'):
    return tf.contrib.layers.max_pool2d(x,k_size,stride=k_stride,padding=padding,scope=scope)


# ------------------------ NETWORK STRUCTURE


def uconv_net(x, training_config, phase, bn_updated_decay = None, verbose = True):
    """
    Create the U-net.
    Input :
        x : TF object to define, ensemble des patchs des images :graph input
        config : dict : described in the header.
        image_size : int : The image size

    Output :
        The U-net.
    """
    
    # Load the variables
    image_size = training_config["trainingset_patchsize"]
    n_classes = training_config["n_classes"]
    depth = training_config["depth"]
    dropout = training_config["dropout"]
    number_of_convolutions_per_layer = training_config["convolution_per_layer"]
    size_of_convolutions_per_layer = training_config["size_of_convolutions_per_layer"]
    features_per_convolution = training_config["features_per_convolution"]
    downsampling = training_config["downsampling"]
    activate_bn = training_config["batch_norm_activate"]
    if bn_updated_decay is None:
        bn_decay = training_config["batch_norm_decay_starting_decay"]
    else:
        bn_decay = bn_updated_decay

    # Input picture shape is [batch_size, height, width, number_channels_in] (number_channels_in = 1 for the input layer)
    net = tf.reshape(x, shape=[-1, image_size, image_size, 1])
    data_temp = x
    data_temp_size = [image_size]
    relu_results = []

    ####################################################################
    ######################### CONTRACTION PHASE ########################
    ####################################################################
    
    for i in range(depth):

        for conv_number in range(number_of_convolutions_per_layer[i]):
            if verbose:
                print(('Layer: ', i, ' Conv: ', conv_number, 'Features: ', features_per_convolution[i][conv_number]))
                print(('Size:', size_of_convolutions_per_layer[i][conv_number]))

            net = conv_relu(net, features_per_convolution[i][conv_number][1], 
                            size_of_convolutions_per_layer[i][conv_number], k_stride=1, 
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout,
                            scope='cconv-d'+str(i)+'-c'+str(conv_number))
                
        relu_results.append(net) # We keep them for the upconvolutions

        if downsampling == 'convolution':
            net = downconv(net, features_per_convolution[i][conv_number][1],  
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay, scope='downconv-d'+str(i))
        else:
            net = maxpool(net, k_size=2, k_stride=2, scope='downmp-d'+str(i))

        data_temp_size.append(data_temp_size[-1] // 2)
        data_temp = net
        
    ####################################################################
    ########################### DEEPEST PHASE ##########################
    ####################################################################

    # For the moment we keem the same number of channels as the last layer we went through
    #net = conv_relu(net, features_per_convolution[i][conv_number][1], 
    #                size_of_convolutions_per_layer[i][conv_number], k_stride=1, 
    #                w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay, scope='deepconv1')
    #
    #net = conv_relu(net, features_per_convolution[i][conv_number][1], 
    #                size_of_convolutions_per_layer[i][conv_number], k_stride=1, 
    #                w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay, scope='deepconv2')
    #
    #data_temp_size.append(data_temp_size[-1])
    #data_temp = net
    
    ####################################################################
    ########################## EXPANSION PHASE #########################
    ####################################################################
    
    for i in range(depth):        
        # Upsampling
        net = tf.image.resize_images(data_temp, [data_temp_size[-1] * 2, data_temp_size[-1] * 2])
        
        # Convolution
        net = conv_relu(net, features_per_convolution[depth - i - 1][-1][1], 
                        k_size=2, k_stride=1, 
                        w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay, scope='upconv-d'+str(depth-i-1))
        
        data_temp_size.append(data_temp_size[-1] * 2)

        # concatenation (see U-net article)
        net = tf.concat(values=[tf.slice(relu_results[depth-i-1], [0, 0, 0, 0], [-1, data_temp_size[depth-i-1],
                                                             data_temp_size[depth-i-1], -1]), net], axis=3)
        
        # Classic convolutions
        for conv_number in range(number_of_convolutions_per_layer[depth - i - 1]):
            if verbose:
                print(('Layer: ', i, ' Conv: ', conv_number, 'Features: ', features_per_convolution[i][conv_number]))
                print(('Size:', size_of_convolutions_per_layer[i][conv_number]))

            net = conv_relu(net, features_per_convolution[depth - i - 1][conv_number][1], 
                            size_of_convolutions_per_layer[depth - i - 1][conv_number], k_stride=1, 
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout,
                            scope='econv-d'+str(depth-i-1)+'-c'+str(conv_number))
        
        data_temp = net

    # Final convolution and segmentation
    
    finalconv = tf.contrib.layers.conv2d(net, num_outputs=n_classes, kernel_size=1, stride=1, scope='finalconv', padding='SAME')
        
    # Adding summary of the activations for the last convolution
    tf.summary.histogram('activations_last', finalconv)
    
    # We also display the weights of the first kernel (can help to detect some mistakes)
    #first_layer_weights_reshaped = visualize_first_layer(weights['wc'][0][0], size_of_convolutions_per_layer[0][0], features_per_convolution[0][0][1])
    #tf.summary.image("Visualize_kernel", first_layer_weights_reshaped)
    
    # Finally we compute the activations of the last layer
    final_result = tf.reshape(finalconv,
        [tf.shape(finalconv)[0], data_temp_size[-1] * data_temp_size[-1], n_classes])

    return final_result