# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import math
import numpy as np
import os
import pickle
from data.input_data import input_data
from config import generate_config


# import matplotlib.pyplot as plt

########## HEADER ##########
# Config file description :

# network_learning_rate : float : No idea, but certainly linked to the back propagation ? Default : 0.0005.

# network_n_classes : int : number of labels in the output. Default : 2.

# network_dropout : float : between 0 and 1 : percentage of neurons we want to keep. Default : 0.75.

# network_depth : int : number of layers. Default : 6.

# network_convolution_per_layer : list of int, length = network_depth : number of convolution per layer. Default : [1 for i in range(network_depth)].

# network_size_of_convolutions_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer]] : Describe the size of each convolution filter.
# Default : [[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)].

# network_features_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer[2]] : Numer of different filters that are going to be used.
# Default : [[64 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]. WARNING ! network_features_per_layer[k][1] = network_features_per_layer[k+1][0].

# network_trainingset : string : describe the trainingset for the network.

# network_downsampling : string 'maxpooling' or 'convolution' : the downsampling method.

# network_thresholds : list of float in [0,1] : the thresholds for the ground truthes labels.

# network_weighted_cost : boolean : whether we use weighted cost for training or not.
###########################


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Compute the weights and biases for the network.
def compute_weights(config):
    """
    Create the weights and biases.
    Input :
        x : TF object to define, ensemble des patchs des images :graph input
        config : dict : described in the header.
        dropout : float between 0 and 1 : percentage of neurons kept, 
        image_size : int : The image size

    Output :
        The U-net.
    """
    image_size = 256
    n_input = image_size * image_size

    learning_rate = config["network_learning_rate"]
    n_classes = config["network_n_classes"]
    dropout = config["network_dropout"]
    depth = config["network_depth"]
    number_of_convolutions_per_layer = config["network_convolution_per_layer"]
    size_of_convolutions_per_layer = config["network_size_of_convolutions_per_layer"]
    features_per_convolution = config["network_features_per_convolution"]
    downsampling = config["network_downsampling"]
    weighted_cost = config["network_weighted_cost"]
    thresh_indices = config["network_thresholds"]

    ####################################################################
    # Create some wrappers for simplicity

    if downsampling == 'convolution':
        weights = {'upconv': [], 'finalconv': [], 'wb1': [], 'wb2': [], 'wc': [], 'we': [], 'pooling': []}
        biases = {'upconv_b': [], 'finalconv_b': [], 'bb1': [], 'bb2': [], 'bc': [], 'be': [], 'pooling_b': []}
    elif downsampling == 'maxpooling':
        weights = {'upconv': [], 'finalconv': [], 'wb1': [], 'wb2': [], 'wc': [], 'we': []}
        biases = {'upconv_b': [], 'finalconv_b': [], 'bb1': [], 'bb2': [], 'bc': [], 'be': []}
    else:
        print('Wrong downsampling method, please use ''maxpooling'' or ''convolution''.')

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

            layer_convolutions_weights.append(
                tf.Variable(tf.random_normal([conv_size, conv_size, num_features_in, num_features[1]],
                                             stddev=math.sqrt(2.0 / (conv_size * conv_size * float(num_features_in)))),
                            name='wc' + str(conv_number + 1) + '1-%s' % i))
            layer_convolutions_biases.append(tf.Variable(tf.random_normal([num_features[1]],
                                                                          stddev=math.sqrt(2.0 / (
                                                                          conv_size * conv_size * float(
                                                                              num_features[1])))),
                                                         name='bc' + str(conv_number + 1) + '1-%s' % i))

            num_features_in = num_features[1]
            

        if downsampling == 'convolution':
            weights_pool = tf.Variable(tf.random_normal([5, 5, num_features_in, num_features_in],
                                                        stddev=math.sqrt(2.0 / (25 * float(num_features_in)))),
                                       name='wb1-%s' % i)
            biases_pool = tf.Variable(
                tf.random_normal([num_features_in], stddev=math.sqrt(2.0 / (25 * float(num_features[1])))),
                name='bc' + str(conv_number + 1) + '1-%s' % i)

        # Store contraction layers weights & biases.
        weights['wc'].append(layer_convolutions_weights)
        biases['bc'].append(layer_convolutions_biases)
        if downsampling == 'convolution':
            weights['pooling'].append(weights_pool)
            biases['pooling_b'].append(biases_pool)

    num_features_b = 2 * num_features_in
    weights['wb1'] = tf.Variable(
        tf.random_normal([3, 3, num_features_in, num_features_b], stddev=math.sqrt(2.0 / (9 * float(num_features_in)))),
        name='wb1-%s' % i)
    weights['wb2'] = tf.Variable(
        tf.random_normal([3, 3, num_features_b, num_features_b], stddev=math.sqrt(2.0 / (9 * float(num_features_b)))),
        name='wb2-%s' % i)
    biases['bb1'] = tf.Variable(tf.random_normal([num_features_b]), name='bb1-%s' % i)
    biases['bb2'] = tf.Variable(tf.random_normal([num_features_b]), name='bb2-%s' % i)

    num_features_in = num_features_b

    # Expansion
    for i in range(depth):
        
        layer_convolutions_weights = []
        layer_convolutions_biases = []

        num_features = features_per_convolution[depth - i - 1][-1]
        
        weights['upconv'].append(
            tf.Variable(tf.random_normal([2, 2, num_features_in, num_features[1]]), name='upconv-%s' % i))
        biases['upconv_b'].append(tf.Variable(tf.random_normal([num_features[1]]), name='bupconv-%s' % i))

        for conv_number in reversed(range(number_of_convolutions_per_layer[depth - i - 1])):
            
            if conv_number == number_of_convolutions_per_layer[depth - i - 1] - 1:
                num_features_in = features_per_convolution[depth - i - 1][-1][1] + num_features[1]
                print('Input features layer : ', num_features_in)

            # We climb the reversed layers 
            conv_size = size_of_convolutions_per_layer[depth - i - 1][conv_number]
            num_features = features_per_convolution[depth - i - 1][conv_number]
            layer_convolutions_weights.append(
                tf.Variable(tf.random_normal([conv_size, conv_size, num_features_in, num_features[1]],
                                             stddev=math.sqrt(2.0 / (conv_size * conv_size * float(num_features_in)))),
                            name='we' + str(conv_number + 1) + '1-%s' % i))
            layer_convolutions_biases.append(tf.Variable(tf.random_normal([num_features[1]],
                                                                          stddev=math.sqrt(2.0 / (
                                                                          conv_size * conv_size * float(
                                                                              num_features[1])))),
                                                         name='be' + str(conv_number + 1) + '1-%s' % i))
            # Actualisation of next convolution's input number.
            num_features_in = num_features[1]

        # Store expansion layers weights & biases.
        weights['we'].append(layer_convolutions_weights)
        biases['be'].append(layer_convolutions_biases)

    weights['finalconv'] = tf.Variable(tf.random_normal([1, 1, num_features_in, n_classes]), name='finalconv-%s' % i)
    biases['finalconv_b'] = tf.Variable(tf.random_normal([n_classes]), name='bfinalconv-%s' % i)

    return weights,biases


# Create model
def uconv_net(x, config, weights, biases, image_size=256):
    """
    Create the U-net.
    Input :
        x : TF object to define, ensemble des patchs des images :graph input
        config : dict : described in the header.
        dropout : float between 0 and 1 : percentage of neurons kept, 
        image_size : int : The image size

    Output :
        The U-net.
    """
    image_size = image_size
    n_classes = config["network_n_classes"]
    depth = config["network_depth"]
    number_of_convolutions_per_layer = config["network_convolution_per_layer"]
    size_of_convolutions_per_layer = config["network_size_of_convolutions_per_layer"]
    features_per_convolution = config["network_features_per_convolution"]
    downsampling = config["network_downsampling"]

    # Input picture shape is [batch_size, height, width, number_channels_in] (number_channels_in = 1 for the input layer)
    x = tf.reshape(x, shape=[-1, image_size, image_size, 1])
    data_temp = x
    data_temp_size = [image_size]
    relu_results = []

    # contraction
    for i in range(depth):

        for conv_number in range(number_of_convolutions_per_layer[i]):
            print('Layer: ', i, ' Conv: ', conv_number, 'Features: ', features_per_convolution[i][conv_number])
            print('Size:', size_of_convolutions_per_layer[i][conv_number])

            if conv_number == 0:
                convolution_c = conv2d(data_temp, weights['wc'][i][conv_number], biases['bc'][i][conv_number])
            else:
                convolution_c = conv2d(convolution_c, weights['wc'][i][conv_number], biases['bc'][i][conv_number])
                
            # Adding summary of the activations for each layer and each convolution (except the downsampling convolution)
            tf.summary.histogram('activations_contrac_d'+str(i)+'_cn'+str(conv_number), convolution_c)
            
            # If it's the first convolution of the first channel/filter, we also keep a summary of the image of the kernel
        
        relu_results.append(convolution_c)

        if downsampling == 'convolution':
            convolution_c = conv2d(convolution_c, weights['pooling'][i], biases['pooling_b'][i], strides=2)
        else:
            convolution_c = maxpool2d(convolution_c, k=2)

        data_temp_size.append(data_temp_size[-1] / 2)
        data_temp = convolution_c

    conv1 = conv2d(data_temp, weights['wb1'], biases['bb1'])
    conv2 = conv2d(conv1, weights['wb2'], biases['bb2'])
    data_temp_size.append(data_temp_size[-1])
    data_temp = conv2
    
    # Adding summary of the activations for the deepest layers
    tf.summary.histogram('activations_deepest_c0', conv1)
    tf.summary.histogram('activations_deepest_c1', conv2)

    # expansion
    for i in range(depth):
        # Resizing + upconvolution
        data_temp = tf.image.resize_images(data_temp, [data_temp_size[-1] * 2, data_temp_size[-1] * 2])
        upconv = conv2d(data_temp, weights['upconv'][i], biases['upconv_b'][i])
        data_temp_size.append(data_temp_size[-1] * 2)

        # concatenation (see U-net article)
        upconv_concat = tf.concat(values=[tf.slice(relu_results[depth - i - 1], [0, 0, 0, 0],
                                                                 [-1, data_temp_size[depth - i - 1],
                                                                  data_temp_size[depth - i - 1], -1]), upconv],
                                  axis=3)
        
        # Classic convolutions
        for conv_number in range(number_of_convolutions_per_layer[i]):
            print('Layer: ', i, ' Conv: ', conv_number, 'Features: ', features_per_convolution[i][conv_number])
            print('Size:', size_of_convolutions_per_layer[i][conv_number])

            if conv_number == 0:
                convolution_e = conv2d(upconv_concat, weights['we'][i][conv_number], biases['be'][i][conv_number])
            else:
                convolution_e = conv2d(convolution_e, weights['we'][i][conv_number], biases['be'][i][conv_number])
                
            # Adding summary of the activations for each layer and each convolution (except the upsampling convolution)
            tf.summary.histogram('activations_expand_d'+str(depth-i-1)+'_cn'+str(conv_number), convolution_e)
        

        data_temp = convolution_e

    # final convolution and segmentation
    finalconv = tf.nn.conv2d(convolution_e, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
    
    # Adding summary of the activations for the last convolution
    tf.summary.histogram('activations_last', finalconv)
    
    # We also display the weights of the first kernel (can help to detect some mistakes)
    first_layer_weights_reshaped = visualize_first_layer(weights['wc'][0][0], image_size, size_of_convolutions_per_layer[0][0], features_per_convolution[0][0][1])
    tf.summary.image("Visualize_kernel", first_layer_weights_reshaped)
    
    # Finally we compute the activations of the last layer
    
    final_result = tf.reshape(finalconv,
        [tf.shape(finalconv)[0], data_temp_size[-1] * data_temp_size[-1], n_classes])

    return final_result
    

def train_model(path_trainingset, path_model, config, path_model_init=None,
                save_trainable=True, augmented_data=True, gpu=None):
    """
    :param path_trainingset: path of the train and validation set built from data_construction
    :param path_model: path to save the trained model
    :param config: dict: network's parameters described in the header.
    :param path_model_init: (option) path of the model to initialize  the training
    :param learning_rate: learning_rate of the optimiser
    :param save_trainable: if True, only weights are saved. If false the variables from the optimisers are saved too
    :param verbose:
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return:
    """

    # Divers variables
    Loss = []
    Epoch = []
    Accuracy = []
    Report = ''
    verbose = 1

    # Results and Models
    folder_model = path_model
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    display_step = 100
    save_step = 600

    # Network Parameters
    image_size = 256
    n_input = image_size * image_size

    learning_rate = config["network_learning_rate"]
    n_classes = config["network_n_classes"]
    dropout = config["network_dropout"]
    depth = config["network_depth"]
    number_of_convolutions_per_layer = config["network_convolution_per_layer"]
    size_of_convolutions_per_layer = config["network_size_of_convolutions_per_layer"]
    features_per_convolution = config["network_features_per_convolution"]
    downsampling = config["network_downsampling"]
    weighted_cost = config["network_weighted_cost"]
    thresh_indices = config["network_thresholds"]

    # ----------------SAVING HYPERPARAMETERS TO USE THEM FOR apply_model-----------------------------------------------#

    hyperparameters = {'depth': depth, 'dropout': dropout, 'image_size': image_size,
                       'model_restored_path': path_model_init, 'learning_rate': learning_rate,
                       'network_n_classes': n_classes, 'network_downsampling': downsampling,
                       'network_thresholds': thresh_indices, 'weighted_cost': weighted_cost,
                       'network_convolution_per_layer': number_of_convolutions_per_layer,
                       'network_size_of_convolutions_per_layer': size_of_convolutions_per_layer,
                       'network_features_per_convolution': features_per_convolution}

    with open(folder_model + '/hyperparameters.pkl', 'wb') as handle:
        pickle.dump(hyperparameters, handle)

    data_train = input_data(trainingset_path=path_trainingset, type='train', thresh_indices=thresh_indices)
    data_validation = input_data(trainingset_path=path_trainingset, type='validation', thresh_indices=thresh_indices)
    
    batch_size_validation = data_validation.get_size()

    # Optimization Parameters
    batch_size = 1
    training_iters = 500000
    epoch_size = data_train.get_size()

    Report += '\n\n---Savings---'
    Report += '\n Model saved in : ' + folder_model

    Report += '\n\n---PARAMETERS---\n'
    Report += 'learning_rate : ' + str(learning_rate) + '; \n batch_size :  ' + str(batch_size) + ';\n depth :  ' + str(
        depth) \
              + ';\n epoch_size: ' + str(epoch_size) + ';\n dropout :  ' + str(dropout) \
              + ';\n (if model restored) restored_model :' + str(path_model_init)


    # Graph input
    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size)) # None should be batch_size
    y = tf.placeholder(tf.float32, shape=(None, image_size, image_size, n_classes)) # Should be batch_size x n_input

    if weighted_cost == True:
        spatial_weights = tf.placeholder(tf.float32, shape=(None, image_size, image_size)) # Should be batch_size x n_input

    keep_prob = tf.placeholder(tf.float32)
    adapt_learning_rate = tf.placeholder(tf.float32)

    weights, biases = compute_weights(config)
    ####################################################

    # Call the model, selected a GPU if asked
    # WARNING : THIS IS FOR BIRELI, THERE ARE ONLY 2 GPUs
    if gpu in ['gpu:0', 'gpu:1']:
        with tf.device('/' + gpu):
            pred = uconv_net(x, config, weights, biases)
    else:
        pred = uconv_net(x, config, weights, biases)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print('tot_param = ',total_parameters)
    
    # Reshaping pred and y so that they are understandable by softmax_cross_entropy
    
    pred_ = tf.reshape(pred, [-1,tf.shape(pred)[-1]], name='Reshape_pred')
    y_ = tf.reshape(tf.reshape(y,[-1,tf.shape(y)[1]*tf.shape(y)[2], tf.shape(y)[-1]]), [-1,tf.shape(y)[-1]], name='Reshape_y')
   
    # Define loss and optimizer
    if weighted_cost == True:
        # Reshaping the weights matrix to a vector of good length
        spatial_weights_ = tf.reshape(tf.reshape(spatial_weights,[-1,tf.shape(spatial_weights)[1]*tf.shape(spatial_weights)[2]]), [-1], name='Reshape_spatial_weights')
        
        cost = tf.reduce_mean(tf.multiply(spatial_weights_,tf.nn.softmax_cross_entropy_with_logits(logits=pred_, labels=y_)))
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_, labels=y_))
    ########

    temp = set(tf.global_variables())  # trick to get variables generated by the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    
    correct_pred = tf.equal(tf.argmax(pred_, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Define variables to keep track of the train error over one whole epoch instead of just one batch (these are the ones we are going to summarize)
    
    L_training_loss = tf.placeholder(tf.float32)
    L_training_acc = tf.placeholder(tf.float32)
    
    training_loss = tf.reduce_mean(L_training_loss)
    training_acc = tf.reduce_mean(tf.cast(L_training_acc, tf.float32))
    
    tf.summary.scalar('loss', training_loss)
    tf.summary.scalar('accuracy', training_acc)
    
    # Merged summaries relates to all numeric data (histograms, kernels ... etc)
    merged_summaries = tf.summary.merge_all()

    
    # We now create a summary specific to images. We add images of the input and its transformation by the u-net
        
    L_im_summ = []
    L_im_summ.append(tf.summary.image('input_image', tf.expand_dims(x, axis = -1)))
    softmax_pred = tf.reshape(tf.reshape(tf.nn.softmax(pred_), (-1, image_size * image_size, n_classes)), (-1, image_size, image_size, n_classes)) # We compute the softmax prediction to display the probability map and reshape them to (b_s, imsz, imsz, n_classes)
    
    probability_maps = tf.split(softmax_pred, n_classes, axis=3)
    
    # We create a probability map for each class
    for i, probmap in enumerate(probability_maps):
        L_im_summ.append(tf.summary.image('probability_map_class_'+str(i), probmap))
    
    # Merging the image summary
    images_merged_summaries = tf.summary.merge(L_im_summ)

    ######## Initializing variables and summaries
    
    # We create the directories where we will store our model

    train_writer = tf.summary.FileWriter(logdir=path_model + '/train')
    validation_writer = tf.summary.FileWriter(logdir=path_model + '/validation')

    init = tf.global_variables_initializer()

    if save_trainable:
        saver = tf.train.Saver(tf.trainable_variables())

    else:
        saver = tf.train.Saver(tf.all_variables())

    # Launch the graph
    Report += '\n\n---Intermediary results---\n'

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        last_epoch = 0
        epoch_training_loss = []
        epoch_training_acc = []

        # Setting the graph in the summaries writer in order to be able to use TensorBoard
        train_writer.add_graph(session.graph)
        validation_writer.add_graph(session.graph)

        if path_model_init: # load a previous session if requested.
            folder_restored_model = path_model_init
            saver.restore(session, folder_restored_model + "/model.ckpt")
            if save_trainable:
                session.run(tf.global_variables_initializer(set(tf.global_variables()) - temp))
            file = open(folder_restored_model + '/evolution.pkl', 'r')
            evolution_restored = pickle.load(file)
            last_epoch = evolution_restored["steps"][-1]

        else:
            session.run(init)
        print 'training start'

        if weighted_cost == True:
            print('Weighted cost selected')
        else:
            print('Default cost selected')

        step = 1
        epoch = 1 + last_epoch

        acc_current_best = 0
        loss_current_best = 10000

        while step * batch_size < training_iters:
            
            # Compute the optimizer at each training iteration
            if weighted_cost == True:
                batch_x, batch_y, weight = data_train.next_batch_WithWeights(batch_size, rnd=True,
                                                                             augmented_data=augmented_data)
                
                # if were just finished an epoch, we summarize the performance of the
                # net on the training set to see it in tensorboard.
                if step % epoch_size == 0:
                    stepcost, stepacc, _ = session.run([cost, accuracy, optimizer], feed_dict={x: batch_x, y: batch_y,
                                               spatial_weights: weight, keep_prob: dropout})
                    # Evaluating the loss and the accuracy for the dataset
                    epoch_training_loss.append(stepcost)
                    epoch_training_acc.append(stepacc)
                                                           
                    # Writing the summary
                    summary, im_summary = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_training_loss, L_training_acc: epoch_training_acc, spatial_weights: weight})
                    train_writer.add_summary(summary, epoch)
                    train_writer.add_summary(im_summary, epoch)
                    
                    epoch_training_loss = []
                    epoch_training_acc = []

                else:
                    
                    stepcost, stepacc, _ = session.run([cost, accuracy, optimizer], feed_dict={x: batch_x, y: batch_y,
                                                       spatial_weights: weight,
                                                       keep_prob: dropout})
                    epoch_training_loss.append(stepcost)
                    epoch_training_acc.append(stepacc)
                    
                 
            else: # no weighted cost
                batch_x, batch_y = data_train.next_batch(batch_size, rnd=True, augmented_data=augmented_data)

                # if were just finished an epoch, we summarize the performance of the
                # net on the training set to see it in tensorboard.
                if step % epoch_size == 0:
                    stepcost, stepacc, _ = session.run([cost, accuracy, optimizer], feed_dict={x: batch_x, y: batch_y,
                                                   keep_prob: dropout})
                    # Evaluating the loss and the accuracy for the dataset
                    epoch_training_loss.append(stepcost)
                    epoch_training_acc.append(stepacc)
                                        
                    # Writing the summary
                    summary, im_summary = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_training_loss, L_training_acc: epoch_training_acc})
                    
                    train_writer.add_summary(summary, epoch)
                    train_writer.add_summary(im_summary, epoch)

                    epoch_training_loss = []
                    epoch_training_acc = []

                else:
                    stepcost, stepacc, _ = session.run([cost, accuracy, optimizer], feed_dict={x: batch_x, y: batch_y,
                                                           keep_prob: dropout})
                    epoch_training_loss.append(stepcost)
                    epoch_training_acc.append(stepacc)

            # Every now and then we display the performance of the network on the training set, on the current batch
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                if weighted_cost == True:
                    loss, acc, p = session.run([cost, accuracy, pred], feed_dict={x: batch_x, y: batch_y,
                                                                               spatial_weights: weight, keep_prob: 1.})
                else:
                    loss, acc, p = session.run([cost, accuracy, pred], feed_dict={x: batch_x, y: batch_y,
                                                                               keep_prob: 1.})
                if verbose == 2:
                    outputs = "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc)
                    print outputs


            # At the end of every epoch we compute the performance of our network on the validation set and we
            # save the summaries to see them on TensorBoard
            if step % epoch_size == 0:

                # We retrieve the validation set, and we compute the loss and accuracy on the whole validation set
                data_validation.set_batch_start()
                if weighted_cost == True:
                    batch_x, batch_y, weight = data_validation.next_batch_WithWeights(data_validation.get_size(), rnd=False,
                                                                                      augmented_data=False)
                    
                    loss, acc = session.run([cost, accuracy],
                                         feed_dict={x: batch_x, y: batch_y, spatial_weights: weight, keep_prob: 1.})
                    # Writing the summary for this step of the training, to use in Tensorflow
                    summary, im_summary_val = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: loss, L_training_acc: acc, spatial_weights: weight})

                else:
                    batch_x, batch_y = data_validation.next_batch(data_validation.get_size(), rnd=False, augmented_data=False)
                    loss, acc = session.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

                   # Writing the summary for this step of the training, to use in Tensorflow
                    summary, im_summary_val = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: loss, L_training_acc: acc})
                    
                    
                validation_writer.add_summary(summary, epoch)
                validation_writer.add_summary(im_summary_val, epoch)
                

                Accuracy.append(acc)
                Loss.append(loss)
                Epoch.append(epoch)

                output_2 = '\n----\n Last epoch: ' + str(epoch)
                output_2 += '\n Accuracy: ' + str(acc) + ';'
                output_2 += '\n Loss: ' + str(loss) + ';'
                print '\n\n----Scores on validation:---' + output_2

                # Saving model if it's the best one

                if epoch == 1:
                    acc_current_best = acc
                    loss_current_best = loss

                    # If new model is better than the last one, update best model
                elif (acc > acc_current_best and loss < loss_current_best):
                    save_path = saver.save(session, folder_model + "/best_model.ckpt")

                epoch += 1

            # Saving the model

            if step % save_step == 0:
                evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
                with open(folder_model + '/evolution.pkl', 'wb') as handle:
                    pickle.dump(evolution, handle)
                save_path = saver.save(session, folder_model + "/model.ckpt")

                print("Model saved in file: %s" % save_path)
                file = open(folder_model + "/report.txt", 'w')
                file.write(Report + output_2)
                file.close()

            step += 1

        save_path = saver.save(session, folder_model + "/model.ckpt")

        # Initialize best model with model after epoch 1

        evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
        with open(folder_model + '/evolution.pkl', 'wb') as handle:
            pickle.dump(evolution, handle)

        print("Model saved in file: %s" % save_path)
        print "Optimization Finished!"
        

def visualize_first_layer(W_conv1, imsize, filter_size, n_filter):
    '''
    :param W_conv1: weights of the first convolution of the first layer
    :param imsize: size of the training images
    :return W1_e: pre-processed data to be added to the summary. Will be used to visualize the kernels of the first layer
    '''
        
    w_display = int(np.ceil(np.sqrt(n_filter)))
    n_filter_completion = int(w_display*w_display - n_filter) # Number of blank filters to add to ensure the display
        
    # Note: we use the dimensions of the current model
    
    W1_a = W_conv1                       # [5, 5, 1, 10]
    W1pad= tf.zeros([filter_size, filter_size, 1, 1])        # [5, 5, 1, 6]  - four zero kernels for padding
    # We have a 4 by 4 grid of kernel visualizations. Therefore, we concatenate 6 empty filters
        
    W1_b = tf.concat([W1_a] + n_filter_completion * [W1pad], axis=3)   # [5, 5, 1, 16]    
    
    W1_c = tf.split(W1_b, w_display*w_display, axis=3 )         # 16 x [5, 5, 1, 1]
    
    # Ici fonction qui ajoute du blanc autour
        
    L_rows = []
    for i in range(w_display):
        L_rows.append(tf.concat(W1_c[0+i*w_display:(i+1)*w_display], axis=0))    # [20, 5, 1, 1] for each element of the list
    W1_d = tf.concat(L_rows, axis=1) # [20, 20, 1, 1]
    W1_e = tf.reshape(W1_d, [1, w_display*filter_size, w_display*filter_size, 1])
    
    return W1_e
        
# To Call the training in the terminal

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path_training", required=True, help="")
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-c", "--config_file", required=False, help="", default="~/.axondeepseg.json")
    ap.add_argument("-m_init", "--path_model_init", required=False, help="")
    ap.add_argument("-gpu", "--GPU", required=False, help="")

    args = vars(ap.parse_args())
    path_training = args["path_training"]
    path_model = args["path_model"]
    path_model_init = args["path_model_init"]
    config_file = args["config_file"]
    gpu = args["GPU"]

    config = generate_config(config_file)

    train_model(path_training, path_model, config, path_model_init, gpu=gpu)


if __name__ == '__main__':
    main()
