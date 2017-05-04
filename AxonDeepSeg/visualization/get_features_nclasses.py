# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import os
import pickle
import numpy as np
import time
from mrf import run_mrf
from scipy import io
from scipy.misc import imread, imsave
from skimage.transform import rescale
from skimage import exposure
from sklearn import preprocessing
from config import*
import json

#Input que l'on veut: depth, number of features per layer, number of convolution per layer, size of convolutions per layer.    n_classes = 2, dropout = 0.75

# Description du fichier config :
    # network_learning_rate : float : No idea, but certainly linked to the back propagation ? Default : 0.0005.
    # network_n_classes : int : number of labels in the output. Default : 2.
    # dropout : float : between 0 and 1 : percentage of neurons we want to keep. Default : 0.75.
    # network_depth : int : number of layers WARNING : factualy, there will be 2*network_depth layers. Default : 6.
    # network_convolution_per_layer : list of int, length = network_depth : number of convolution per layer. Default : [1 for i in range(network_depth)].
    # network_size_of_convolutions_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer]] : Describe the size of each convolution filter. 
    # Default : [[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)].
    
    # network_features_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer[2]] : Numer of different filters that are going to be used.
    # Default : [[[64,64] for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]. WARNING ! network_features_per_layer[k][1] = network_features_per_layer[k+1][0].
    
    
def im2patches(img, size=256):
    """
    :param img: image to segment.
    :param size: size of the patches to extract (must be the same as used in the learning)
    :return: [img, patches, positions of the patches]
    """

    h, w = img.shape

    if (h == 256 and w == 256):
        patch = img
        patch = exposure.equalize_hist(patch)
        patch = (patch - np.mean(patch)) / np.std(patch)
        positions = [[0,0]]
        patches = [patch]

    else :
        q_h, r_h = divmod(h, size)
        q_w, r_w = divmod(w, size)


        r2_h = size-r_h
        r2_w = size-r_w
        q2_h = q_h + 1
        q2_w = q_w + 1

        q3_h, r3_h = divmod(r2_h, q_h)
        q3_w, r3_w = divmod(r2_w, q_w)

        dataset = []
        positions=[]
        pos = 0
        while pos+size<=h:
            pos2 = 0
            while pos2+size<=w:
                patch = img[pos:pos+size, pos2:pos2+size]
                patch = exposure.equalize_hist(patch)
                patch = (patch - np.mean(patch))/np.std(patch)

                dataset.append(patch)
                positions.append([pos,pos2])
                pos2 = size + pos2 - q3_w
                if pos2 + size > w :
                    pos2 = pos2 - r3_w

            pos = size + pos - q3_h
            if pos + size > h:
                pos = pos - r3_h

        patches = np.asarray(dataset)
    return [img, patches, positions]


def patches2im(predictions, positions, image_height, image_width):
    """
    :param predictions: list of the segmentation masks on the patches
    :param positions: positions of the segmentations masks
    :param h_size: height of the image to reconstruct
    :param w_size: width of the image to reconstruct
    :return: reconstructed segmentation on the full image from the masks and their positions
    """
    image = np.zeros((image_height, image_width))
    for pred, pos in zip(predictions, positions):
        image[pos[0]:pos[0]+256, pos[1]:pos[1]+256] = pred
    return image
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def Uconv_net(x, config, dropout, image_size = 256, target_features = [0,0]):
    """
    Create the U-net.
    Input :
           x : TF object to define, ensemble des patchs des images :graph input

           weights['wc'] : list of lists containing the convolutions' weights of the contraction layers.
           weights['we'] : list of lists containing the convolutions' weights of the expansion layers.
           biases['bc'] : list of lists containing the convolutions' biases of the contraction layers.
           biases['be'] : list of lists containing the convolutions' biases of the expansion layers.
           
           weights['wb'] : list of the bottom layer's convolutions' weights.
           biases['bb'] : list of the bottom layer's convolutions' biases.

           weights['upconv'] : list of the upconvolutions layers convolutions' weights.
           biases['upconv_b'] : list of the upconvolutions layers convolutions' biases.

           weights['finalconv'] : list of the last layer convolutions' weights.
           biases['finalconv_b'] : list of the last layer convolutions' biases.

           dropout : float between 0 and 1 : percentage of neurons kept, 
           image_size : int : The image size

    Output :
           The U-net.
    """
    
    image_size = 256
    n_input = image_size * image_size
    learning_rate = config.get("network_learning_rate", 0.0005)
    n_classes = config.get("network_n_classes", 2)
    dropout = config.get("network_dropout", 0.75)
    depth = config.get("network_depth", 6)
    number_of_convolutions_per_layer = config.get("network_convolution_per_layer", [1 for i in range(depth)])
    size_of_convolutions_per_layer =  config.get("network_size_of_convolutions_per_layer",[[3 for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])
    features_per_convolution = config.get("network_features_per_convolution",[[[64,64] for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])

####################################################################
    # Create some wrappers for simplicity
    
    weights = {'upconv':[],'finalconv':[],'wb1':[], 'wb2':[], 'wc':[], 'we':[],'pooling':[]}
    biases = {'upconv_b':[],'finalconv_b':[],'bb1':[], 'bb2':[], 'bc':[], 'be':[],'pooling_b':[]}                                  
     

    # Contraction
    for i in range(depth):

        layer_convolutions_weights = []
        layer_convolutions_biases = []

        # Compute the layer's convolutions and biases.
        for conv_number in range(number_of_convolutions_per_layer[i]):

            conv_size = size_of_convolutions_per_layer[i][conv_number]
            num_features = features_per_convolution[i][conv_number]

            # Use 1 if it is the first convolution : input.
            if i == 0 and conv_number == 0:
                num_features_in = 1

            layer_convolutions_weights.append(tf.Variable(tf.random_normal([conv_size, conv_size, num_features_in, num_features[1]],
                                                                stddev=math.sqrt(2.0/(conv_size*conv_size*float(num_features_in)))), name = 'wc'+str(conv_number+1)+'1-%s'%i))
            layer_convolutions_biases.append(tf.Variable(tf.random_normal([num_features[1]],
                                                                    stddev=math.sqrt(2.0/(conv_size*conv_size*float(num_features[1])))), name='bc'+str(conv_number+1)+'1-%s'%i))
            
            num_features_in = num_features[1]

        weights_pool = tf.Variable(tf.random_normal([5, 5, num_features_in, num_features_in], stddev=math.sqrt(2.0/(25*float(num_features_in)))),name='wb1-%s'%i)
        biases_pool = tf.Variable(tf.random_normal([num_features_in], stddev=math.sqrt(2.0/(25*float(num_features[1])))), name='bc'+str(conv_number+1)+'1-%s'%i)
        # Store contraction layers weights & biases.
        weights['wc'].append(layer_convolutions_weights)
        biases['bc'].append(layer_convolutions_biases)
        weights['pooling'].append(weights_pool)
        biases['pooling_b'].append(biases_pool)

    num_features_b = 2*num_features_in
    weights['wb1'] = tf.Variable(tf.random_normal([3, 3, num_features_in, num_features_b], stddev=math.sqrt(2.0/(9*float(num_features_in)))),name='wb1-%s'%i)
    weights['wb2'] = tf.Variable(tf.random_normal([3, 3, num_features_b, num_features_b], stddev=math.sqrt(2.0/(9*float(num_features_b)))), name='wb2-%s'%i)
    biases['bb1'] = tf.Variable(tf.random_normal([num_features_b]), name='bb1-%s'%i)
    biases['bb2'] = tf.Variable(tf.random_normal([num_features_b]), name='bb2-%s'%i)

    num_features_in = num_features_b

    # Expansion
    for i in range(depth):


        layer_convolutions_weights = []
        layer_convolutions_biases = []

        num_features = features_per_convolution[depth-i-1][-1]

        weights['upconv'].append(tf.Variable(tf.random_normal([2, 2, num_features_in, num_features[1]]), name='upconv-%s'%i))
        biases['upconv_b'].append(tf.Variable(tf.random_normal([num_features[1]]), name='bupconv-%s'%i))

        for conv_number in reversed(range(number_of_convolutions_per_layer[depth-i-1])):

            if conv_number == number_of_convolutions_per_layer[depth-i-1]-1:
                num_features_in = features_per_convolution[depth-i-1][-1][1]+num_features[1]
                print('Input features layer : ',num_features_in)

            # We climb the reversed layers 
            conv_size = size_of_convolutions_per_layer[depth-i-1][conv_number]
            num_features = features_per_convolution[depth-i-1][conv_number]
            # print(num_features[1])
            layer_convolutions_weights.append(tf.Variable(tf.random_normal([conv_size,conv_size, num_features_in, num_features[1]],
                                                                    stddev=math.sqrt(2.0/(conv_size*conv_size*float(num_features_in)))), name = 'we'+str(conv_number+1)+'1-%s'%i))
            layer_convolutions_biases.append(tf.Variable(tf.random_normal([num_features[1]],
                                                                        stddev=math.sqrt(2.0/(conv_size*conv_size*float(num_features[1])))), name='be'+str(conv_number+1)+'1-%s'%i))
            # Actualisation of next convolution's input number.
            num_features_in = num_features[1]

        # Store expansion layers weights & biases.
        weights['we'].append(layer_convolutions_weights)
        biases['be'].append(layer_convolutions_biases)

    weights['finalconv']= tf.Variable(tf.random_normal([1, 1, num_features_in, n_classes]), name='finalconv-%s'%i)
    biases['finalconv_b']= tf.Variable(tf.random_normal([n_classes]), name='bfinalconv-%s'%i)
    ####################################################

# Reshape input picture
    
    x = tf.reshape(x, shape=[-1, image_size, image_size, 1])
    data_temp = x
    data_temp_size = [image_size]
    relu_results = []

    limit_to_one_image = 0
    limit_to_one_image_end = 0
    # contraction
    for i in range(depth):

        for conv_number in range(number_of_convolutions_per_layer[i]):
            print('Layer: ',i,' Conv: ',conv_number, 'Features: ', features_per_convolution[i][conv_number])
            print('Size:',size_of_convolutions_per_layer[i][conv_number])
            if conv_number == target_features[1]:
                convolution_c = conv2d(data_temp, weights['wc'][i][conv_number], biases['bc'][i][conv_number])
                if i == target_features[0]:
                    with tf.name_scope('convolution'):

                        if limit_to_one_image == 0:
                            for iteration in range(convolution_c.get_shape().as_list()[-1]-1):
                                tf.image_summary("Visualize_image_number_"+str(iteration), convolution_c[:,:,:,iteration:iteration+1])
                            limit_to_one_image +=1

                        channels = features_per_convolution[target_features[0]][target_features[1]][1]
                        conv_size = size_of_convolutions_per_layer[target_features[0]][target_features[1]]

                        W_a = weights['wc'][i][conv_number]
                        Wpad= tf.zeros([conv_size, conv_size, 1, 1])
                        # We have a 6 by 6 grid of kernepl visualizations. yet we only have 32 filters
                        # Therefore, we concatenate 4 empty filters
                        W_b = tf.concat(3, [W_a, Wpad]) 
                        
                        for repetition in range(36-channels-1):
                            W_b = tf.concat(3, [W_b, Wpad])   # [5, 5, 1, 36]  

                        W_c = tf.split(3, 36, W_b)         # 36 x [5, 5, 1, 1]
                        W_row0 = tf.concat(0, W_c[0:6])    # [30, 5, 1, 1]
                        W_row1 = tf.concat(0, W_c[6:12])   # [30, 5, 1, 1]
                        W_row2 = tf.concat(0, W_c[12:18])  # [30, 5, 1, 1]
                        W_row3 = tf.concat(0, W_c[18:24])  # [30, 5, 1, 1]
                        W_row4 = tf.concat(0, W_c[24:30])  # [30, 5, 1, 1]
                        W_row5 = tf.concat(0, W_c[30:36])  # [30, 5, 1, 1]
                        W_d = tf.concat(1, [W_row0, W_row1, W_row2, W_row3, W_row4, W_row5]) # [30, 30, 1, 1]
                        W_e = tf.reshape(W_d, [1, 6*conv_size, 6*conv_size, 1])
                        Wtag = tf.placeholder(tf.string, None)
                        tf.image_summary("Visualize_kernels_"+str(conv_number), W_e)


            else:
                convolution_c = conv2d(convolution_c, weights['wc'][i][conv_number], biases['bc'][i][conv_number])

        relu_results.append(convolution_c)
        convolution_c = maxpool2d(convolution_c, k=2)
        data_temp_size.append(data_temp_size[-1]/2)
        data_temp = convolution_c

    conv1 = conv2d(data_temp, weights['wb1'], biases['bb1'])
    conv2 = conv2d(conv1, weights['wb2'], biases['bb2'])
    data_temp_size.append(data_temp_size[-1])
    data_temp = conv2

    # expansion
    for i in range(depth):
        data_temp = tf.image.resize_images(data_temp, [data_temp_size[-1] * 2, data_temp_size[-1] * 2])
        upconv = conv2d(data_temp, weights['upconv'][i], biases['upconv_b'][i])
        data_temp_size.append(data_temp_size[-1]*2)

        # concatenation
        upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[depth-i-1], [0, 0, 0, 0],
                                                                 [-1, data_temp_size[depth-i-1], data_temp_size[depth-i-1], -1]), upconv])
        
        for conv_number in range(number_of_convolutions_per_layer[i]):
            print('Layer: ',i,' Conv: ',conv_number, 'Features: ', features_per_convolution[i][conv_number])
            if conv_number == 0:
                convolution_e = conv2d(upconv_concat, weights['we'][i][conv_number], biases['be'][i][conv_number])
            else:
                convolution_e = conv2d(convolution_e, weights['we'][i][conv_number], biases['be'][i][conv_number])

        data_temp = convolution_e

    if limit_to_one_image_end == 0:
        for iteration in range(convolution_e.get_shape().as_list()[-1]-1):
            tf.image_summary("Visualize_image_end_"+str(iteration), convolution_e[:,:,:,iteration:iteration+1])
        limit_to_one_image_end +=1

    # final convolution and segmentation
    finalconv = tf.nn.conv2d(convolution_e, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
    final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], n_classes]))

    return final_result

def get_convnet_features(path_my_data, path_model, config, folder_write = None, target_features = [0,0], thresh_indices = [0,0.5]):
    """
    :param target_features: list of two int: the convolution target we want to get the output.
    :param config: json file, description on file header.
    :param path_my_data: folder of the image to segment. Must contain image.jpg
    :param path_model: folder of the model of segmentation. Must contain model.ckpt
    :return: prediction, the mask of the segmentation
    """
    if folder_write == None:
        folder_write = path_model

    print '\n\n ---Start axon segmentation on %s---' % path_my_data

    path_img = path_my_data + '/image.jpg'
    img = imread(path_img, flatten=False, mode='L')

    file = open(path_my_data + '/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())

    # set the resolution to the general_pixel_size
    rescale_coeff = pixel_size/general_pixel_size
    img = (rescale(img, rescale_coeff)*256).astype(int)

    batch_size = 1

    ###############
    # Network Parameters
    image_size = 256
    n_input = image_size * image_size
    learning_rate = config.get("network_learning_rate", 0.0005)
    n_classes = config.get("network_n_classes", 2)
    dropout = config.get("network_dropout", 0.75)
    depth = config.get("network_depth", 6)
    number_of_convolutions_per_layer = config.get("network_convolution_per_layer", [1 for i in range(depth)])
    size_of_convolutions_per_layer =  config.get("network_size_of_convolutions_per_layer",[[3 for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])
    features_per_convolution = config.get("network_features_per_convolution",[[[64,64] for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])
    ##############

    folder_model = path_model
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    if os.path.exists(folder_model+'/hyperparameters.pkl'):
        print 'hyperparameters detected in the model'
        hyperparameters = pickle.load(open(folder_model +'/hyperparameters.pkl', "rb"))
        depth = hyperparameters['depth']
        image_size = hyperparameters['image_size']

    #--------------------SAME ALGORITHM IN TRAIN_model---------------------------

    x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
    y = tf.placeholder(tf.float32, shape=(batch_size*n_input, n_classes))
    keep_prob = tf.placeholder(tf.float32)

    # Construct model
    pred = Uconv_net(x, config, keep_prob, target_features = target_features)

    saver = tf.train.Saver(tf.all_variables())

    summary = tf.merge_all_summaries()
    # Image to batch
    image_init, data, positions = im2patches(img, 256)
    predictions = []

    # Launch the graph
    sess = tf.Session()
    saver.restore(sess, folder_model+ '/model.ckpt')

    writer = tf.train.SummaryWriter(folder_write)

    #--------------------- Apply the segmentation to each patch of the images--------------------------------

    for i in range(len(data)):
        print 'processing patch %s on %s'%(i+1, len(data))
        batch_x = np.asarray([data[i]])
        start = time.time()
        p,summary_str = sess.run([pred,summary], feed_dict={x: batch_x})
        Mask = np.zeros_like(p[:,0])
        for pixel in range(len(p[:,0])):
            Mask[pixel] = np.argmax(p[pixel,:])
        
        Mask.reshape(256,256)
        predictions.append(Mask)

        writer.add_summary(summary_str, i)

    writer.flush()
    writer.close()
    sess.close()
