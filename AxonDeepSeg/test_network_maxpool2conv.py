# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import numpy as np
import os
import pickle
import time
from learning.input_data import input_data
from skimage.util import random_noise
import sys

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
    # Default : [[64 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]. WARNING ! network_features_per_layer[k][1] = network_features_per_layer[k+1][0].
    
def gaussian_noise(image):
    """
    :param image: input image 256*256 for the network.
    :return: application of the gaussian noise to the image
    """
    img = image
    noisy_img = random_noise(img, mode='gaussian')
    return noisy_img

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def Uconv_net(x, config, dropout, image_size = 256):
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

    # contraction
    for i in range(depth):

        for conv_number in range(number_of_convolutions_per_layer[i]):
            print('Layer: ',i,' Conv: ',conv_number, 'Features: ', features_per_convolution[i][conv_number])
            print('Size:',size_of_convolutions_per_layer[i][conv_number])

            if conv_number == 0:
                convolution_c = conv2d(data_temp, weights['wc'][i][conv_number], biases['bc'][i][conv_number])
            else:
                convolution_c = conv2d(convolution_c, weights['wc'][i][conv_number], biases['bc'][i][conv_number])

        relu_results.append(convolution_c)
        convolution_c = conv2d(convolution_c, weights['pooling'][i], biases['pooling_b'][i], strides = 2)
        # convolution_c = maxpool2d(convolution_c, k=2)
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
            print('Size:',size_of_convolutions_per_layer[i][conv_number])
            
            if conv_number == 0:
                convolution_e = conv2d(upconv_concat, weights['we'][i][conv_number], biases['be'][i][conv_number])
            else:
                convolution_e = conv2d(convolution_e, weights['we'][i][conv_number], biases['be'][i][conv_number])

        data_temp = convolution_e

    # final convolution and segmentation
    finalconv = tf.nn.conv2d(convolution_e, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
    final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], 2]))

    return final_result


def train_model(path_trainingset, path_model, config, path_model_init = None, save_trainable = True, verbose = 1,
                augmented_data = True, with_noise = False):
    """
    :param path_trainingset: path of the train and test set built from data_construction
    :param path_model: path to save the trained model
    :param config: json file: network's parameters 
    :param path_model_init: (option) path of the model to initialize  the training
    :param learning_rate: learning_rate of the optimiser
    :param save_trainable: if True, only weights are saved. If false the variables from the optimisers are saved too
    :param verbose:
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
    learning_rate = config.get("network_learning_rate", 0.0005)
    n_classes = config.get("network_n_classes", 2)
    dropout = config.get("network_dropout", 0.75)
    depth = config.get("network_depth", 6)
    number_of_convolutions_per_layer = config.get("network_convolution_per_layer", [1 for i in range(depth)])
    size_of_convolutions_per_layer =  config.get("network_size_of_convolutions_per_layer",[[3 for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])
    features_per_convolution = config.get("network_features_per_convolution",[[[64,64] for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])

    #----------------SAVING HYPERPARAMETERS TO USE THEM FOR apply_model-----------------------------------------------#

    hyperparameters = {'depth': depth,'dropout': dropout, 'image_size': image_size,
                       'model_restored_path': path_model_init, 'learning_rate': learning_rate,
                        'network_n_classes': n_classes,
                        'network_convolution_per_layer': number_of_convolutions_per_layer,
                        'network_size_of_convolutions_per_layer': size_of_convolutions_per_layer,
                        'network_features_per_convolution': features_per_convolution
}

    with open(folder_model+'/hyperparameters.pkl', 'wb') as handle :
            pickle.dump(hyperparameters, handle)

    # Optimization Parameters
    batch_size = 1
    training_iters = 500000
    epoch_size = 200

    Report += '\n\n---Savings---'
    Report += '\n Model saved in : '+ folder_model


    Report += '\n\n---PARAMETERS---\n'
    Report += 'learning_rate : '+ str(learning_rate)+'; \n batch_size :  ' + str(batch_size) +';\n depth :  ' + str(depth) \
            +';\n epoch_size: ' + str(epoch_size)+';\n dropout :  ' + str(dropout)\
            +';\n (if model restored) restored_model :' + str(path_model_init)

    data_train = input_data(trainingset_path=path_trainingset, type='train')
    data_test = input_data(trainingset_path=path_trainingset, type='test')

    # Graph input
    x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
    y = tf.placeholder(tf.float32, shape=(batch_size*n_input, n_classes))
    keep_prob = tf.placeholder(tf.float32)


    # Call the model
    pred = Uconv_net(x, config, keep_prob, image_size = image_size)

    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    tf.scalar_summary('Loss', cost)

    temp = set(tf.all_variables()) # trick to get variables generated by the optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.initialize_all_variables()

    if save_trainable :
        saver = tf.train.Saver(tf.trainable_variables())

    else :
        saver = tf.train.Saver(tf.all_variables())

    # Launch the graph
    Report += '\n\n---Intermediary results---\n'

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        last_epoch = 0
        if path_model_init:
            folder_restored_model = path_model_init
            saver.restore(sess, folder_restored_model+"/model.ckpt")

            if save_trainable :
                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

            file = open(folder_restored_model+'/evolution.pkl','r')
            evolution_restored = pickle.load(file)
            last_epoch = evolution_restored["steps"][-1]

        else:
            sess.run(init)
        print 'training start'

        step = 1
        epoch = 1 + last_epoch

        while step * batch_size < training_iters:
            batch_x, batch_y = data_train.next_batch(batch_size, rnd = True, augmented_data= augmented_data)
            
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            """if with_noise:
                noisy_x = gaussian_noise(batch_x)

                sess.run(optimizer, feed_dict={x: noisy_x, y: batch_y,
                                           keep_prob: dropout})"""


            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc, p = sess.run([cost, accuracy, pred], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                """if with_noise:
                    loss, acc, p = sess.run([cost, accuracy, pred], feed_dict={x: noisy_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})"""

                if verbose == 2:
                    outputs = "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc)
                    print outputs

            if step % epoch_size == 0 :
                start = time.time()
                A = [] # list of accuracy scores on the datatest
                L = [] # list of the Loss, or cost, scores on the dataset

                data_test.set_batch_start()
                for i in range(data_test.set_size):
                    batch_x, batch_y = data_test.next_batch(batch_size, rnd=False, augmented_data= augmented_data)
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

                    A.append(acc)
                    L.append(loss)

                    """if with_noise:
                        noisy_x = gaussian_noise(batch_x)
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                        A.append(acc)
                        L.append(loss)"""

                    if verbose >= 1:
                        print '--\nAccuracy on patch'+str(i)+': '+str(acc)
                        print 'Loss on patch'+str(i)+': '+str(loss)
                Accuracy.append(np.mean(A))
                Loss.append(np.mean(L))
                Epoch.append(epoch)

                output_2 = '\n----\n Last epoch: ' + str(epoch)
                output_2+= '\n Accuracy: ' + str(np.mean(A))+';'
                output_2+= '\n Loss: ' + str(np.mean(L))+';'
                print '\n\n----Scores on test:---' + output_2
                epoch+=1

            if step % save_step == 0:
                evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
                with open(folder_model+'/evolution.pkl', 'wb') as handle:
                    pickle.dump(evolution, handle)
                save_path = saver.save(sess, folder_model+"/model.ckpt")

                print("Model saved in file: %s" % save_path)
                file = open(folder_model+"/report.txt", 'w')
                file.write(Report + output_2)
                file.close()

            step += 1

        save_path = saver.save(sess, folder_model+"/model.ckpt")

        evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
        with open(folder_model+'/evolution.pkl', 'wb') as handle :
            pickle.dump(evolution, handle)

        print("Model saved in file: %s" % save_path)
        print "Optimization Finished!"

# To Call the training in the terminal

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path_training", required=True, help="")
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-m_init", "--path_model_init", required=False, help="")
    ap.add_argument("-lr", "--learning_rate", required=False, help="")
    ap.add_argument("-c", "--config_file", required=True,help="")

    args = vars(ap.parse_args())
    path_training = args["path_training"]
    path_model = args["path_model"]
    path_model_init = args["path_model_init"]
    config_file = args["config_file"]
    learning_rate = args["learning_rate"]
    if learning_rate :
        learning_rate = float(args["learning_rate"])
    else :
        learning_rate = None
        
    with open(config_file, 'r') as fd:
        config= json.loads(fd.read())

    train_model(path_training, path_model, config, path_model_init, learning_rate)
 