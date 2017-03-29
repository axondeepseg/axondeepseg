# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle
import time
import json
from hyperopt import fmin, hp, STATUS_OK, Trials
from learning.input_data import input_data
from skimage import exposure

#Input que l'on veut: depth, number of features per layer, number of convolution per layer, size of convolutions per layer.    n_classes = 2, dropout = 0.75

# Description du fichier config :
    # network_image_size : int : size of the patches 256,
    # network_learning_rate : float : No idea, but certainly linked to the back propagation ? Default : 0.0005.
    # network_n_classes : int : number of labels in the output. Default : 2.
    # dropout : float : between 0 and 1 : percentage of neurons we want to keep. Default : 0.75.
    # network_depth : int : number of layers WARNING : factualy, there will be 2*network_depth layers. Default : 6.
    # network_convolution_per_layer : list of int, length = network_depth : number of convolution per layer. Default : [1 for i in range(network_depth)].
    # network_size_of_convolutions_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer]] : Describe the size of each convolution filter. 
    # Default : [[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)].
    
    # network_features_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer[2]] : Numer of different filters that are going to be used.
    # Default : [[64 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]. WARNING ! network_features_per_layer[k][1] = network_features_per_layer[k+1][0].

def progress(stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = str(stats['n_train']) + " train samples (" + str(stats['n_train_pos']) + " positive)\n"
    s += str(stats['n_test']) + " test samples (" + str(stats['n_test_pos']) + " positive)\n"
    s += "accuracy: " + str(stats['accuracy']) + "\n"
    s += "precision: " + str(stats['precision']) + "\n"
    s += "recall: " + str(stats['recall']) + "\n"
    s += "roc: " + str(stats['roc']) + "\n"
    s += "in " + str(duration) + "s (" + str(stats['n_train'] / duration) + " samples/sec)"
    return s

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

# Create some wrappers for simplicity
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
def Uconv_net(config, x):
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
    # Network Parameters
    image_size = config.get("network_image_size", 0.0005)
    n_input = image_size * image_size
    learning_rate = config.get("network_learning_rate", 0.0005)
    n_classes = config.get("network_n_classes", 2)
    dropout = config.get("network_dropout", 0.75)
    depth = config.get("network_depth", 6)
    number_of_convolutions_per_layer = config.get("network_convolution_per_layer", [1 for i in range(depth)])
    size_of_convolutions_per_layer =  config.get("network_size_of_convolution_per_layer",[[3 for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])
    features_per_convolution = config.get("network_features_per_convolution",[[[64,64] for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])


    ############# WEIGHTS AND BIASES ##############
    weights = {'upconv': [], 'finalconv': [], 'wb1': [], 'wb2': [], 'wc': [], 'we': []}
    biases = {'upconv_b': [], 'finalconv_b': [], 'bb1': [], 'bb2': [], 'bc': [], 'be': []}

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

        # Store contraction layers weights & biases.
        weights['wc'].append(layer_convolutions_weights)
        biases['bc'].append(layer_convolutions_biases)

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
            # print(num_features[1])
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

    ############# Unet BUILDING ##############
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, image_size, image_size, 1])
    data_temp = x
    data_temp_size = [image_size]
    relu_results = []

    # contraction
    for i in range(depth):

        for conv_number in range(number_of_convolutions_per_layer[i]):
            print('Layer: ', i, ' Conv: ', conv_number, 'Features: ', features_per_convolution[i][conv_number])
            if conv_number == 0:
                convolution_c = conv2d(data_temp, weights['wc'][i][conv_number], biases['bc'][i][conv_number])
            else:
                convolution_c = conv2d(convolution_c, weights['wc'][i][conv_number], biases['bc'][i][conv_number])

        relu_results.append(convolution_c)
        convolution_c = maxpool2d(convolution_c, k=2)
        data_temp_size.append(data_temp_size[-1] / 2)
        data_temp = convolution_c

    conv1 = conv2d(data_temp, weights['wb1'], biases['bb1'])
    conv2 = conv2d(conv1, weights['wb2'], biases['bb2'])
    data_temp_size.append(data_temp_size[-1])
    data_temp = conv2

    # expansion
    for i in range(depth):
        data_temp = tf.image.resize_images(data_temp, [data_temp_size[-1] * 2, data_temp_size[-1] * 2])
        upconv = conv2d(data_temp, weights['upconv'][i], biases['upconv_b'][i])
        data_temp_size.append(data_temp_size[-1] * 2)

        # concatenation
        upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[depth - i - 1], [0, 0, 0, 0],
                                                                 [-1, data_temp_size[depth - i - 1],
                                                                  data_temp_size[depth - i - 1], -1]), upconv])

        for conv_number in range(number_of_convolutions_per_layer[i]):
            print('Layer: ', i, ' Conv: ', conv_number, 'Features: ', features_per_convolution[i][conv_number])
            if conv_number == 0:
                convolution_e = conv2d(upconv_concat, weights['we'][i][conv_number], biases['be'][i][conv_number])
            else:
                convolution_e = conv2d(convolution_e, weights['we'][i][conv_number], biases['be'][i][conv_number])

        data_temp = convolution_e

    # final convolution and segmentation
    finalconv = tf.nn.conv2d(convolution_e, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
    final_result = tf.reshape(finalconv, tf.TensorShape(
        [finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], 2]))

    return final_result

class Trainer():

    def __init__(self, path_trainingset, network_model, param_training):
    ###############################################################################################################
    #
    # TODO:  hyperopt_train_test: To be adapted to CNN architecture: https://github.com/fchollet/keras/issues/1591
    #
    ###############################################################################################################

        self.data_train = input_data(trainingset_path=path_trainingset, type='train')
        self.data_test = input_data(trainingset_path=path_trainingset, type='test')
        self.results_path = param_training['results_path']

        # Results and Models
        folder_model = self.results_path
        if not os.path.exists(folder_model):
            os.makedirs(folder_model)

        self.model_name = network_model['model_name']
        self.n_iter = param_training['hyperopt']['number_of_epochs']
        self.network_model = network_model
        self.param_training = param_training


    def hyperparam_optimization(self):

        if 'model_hyperparam' in self.network_model:
            self.model_hyperparam = self.network_model['model_hyperparam']
        else:
            self.model_hyperparam = None

        self.param_hyperopt = self.param_training['hyperopt']

        # hyperparam dict must be provided
        if self.model_hyperparam is not None:
            # Create hyperopt dict compatible with hyperopt Lib
            model_hyperparam_hyperopt = {}
            for param in self.model_hyperparam:
                param_cur = self.model_hyperparam[param]
                model_hyperparam_hyperopt[param] = hp.choice(param, param_cur)


            def hyperparam_train_test(self, params, param_training, path_model_init = None, save_trainable = True, verbose = 1):
                """
                :param path_trainingset: path of the train and test set built from data_construction
                :param path_model: path to save the trained model
                :param config: dict: network's parameters
                :param path_model_init: (option) path of the model to initialize  the training
                :param learning_rate: learning_rate of the optimiser
                :param save_trainable: if True, only weights are saved. If false the variables from the optimisers are saved too
                :param verbose:
                :return:
                """

                def results_stats(self, session, fname_out_list, stats = None):
                    ###############################################################################################################
                    #
                    # session :
                    # return : pred and true labels
                    #
                    ###############################################################################################################


                    if stats is None:
                        stats = {'n_train': 0, 'n_train_pos': 0,
                                 'n_test': 0, 'n_test_pos': 0,
                                 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'roc': 0.0,
                                 'accuracy_history': [(0, 0)], 'precision_history': [(0, 0)], 'recall_history': [(0, 0)],
                                 'roc_history': [(0, 0)],
                                 't0': time.time(), 'total_fit_time': 0.0}

                    stats['prediction_time'] = 0
                    y_pred, y_test = [], []

                    self.data_test.set_batch_start()
                    for i in range(self.data_test.set_size):
                        X_test, y_test_cur = self.data_test.next_batch(batch_size, rnd=False, augmented_data=False)
                        y_pred_cur = session.run(y_pred_cur, feed_dict={x: X_test, y: y_test_cur, keep_prob: 1.})

                        tick = time.time()

                        stats['prediction_time'] += time.time() - tick
                        y_pred.extend(y_pred_cur)
                        y_test.extend(y_test_cur)
                        stats['n_test'] += X_test.shape[0]
                        stats['n_test_pos'] += sum(y_test_cur)

                    y_test = np.array(y_test)
                    y_pred = np.array(y_pred)
                    stats['accuracy'] = accuracy_score(y_test, y_pred)
                    stats['precision'] = precision_score(y_test, y_pred)
                    stats['recall'] = recall_score(y_test, y_pred)
                    stats['roc'] = roc_auc_score(y_test, y_pred)

                    acc_history = (stats['accuracy'], stats['n_train'])
                    stats['accuracy_history'].append(acc_history)

                    precision_history = (stats['precision'], stats['n_train'])
                    stats['precision_history'].append(precision_history)

                    recall_history = (stats['recall'], stats['n_train'])
                    stats['recall_history'].append(recall_history)

                    roc_history = (stats['roc'], stats['n_train'])
                    stats['roc_history'].append(roc_history)

                    print progress(stats)

                    pickle.dump(stats, open(
                        self.results_path + self.model_name + '_eval_' + str(fname_out_list[0]).zfill(12) + '_' + str(
                            fname_out_list[1]).zfill(6) + '.pkl', "wb"))

                    return y_test, y_pred

                # Divers variables
                Loss = []
                Epoch = []
                Accuracy = []
                Report = ''
                verbose = 1

                # Results and Models
                folder_model = self.results_path
                if not os.path.exists(folder_model):
                    os.makedirs(folder_model)

                display_step = 100
                save_step = 600

                # For hyperopt
                param_training = param_training
                param_hyperopt = param_training['hyperopt']

                # Network Parameters
                image_size = params.get("network_image_size", 0.0005)
                n_input = image_size * image_size
                n_classes = params.get("network_n_classes", 2)
                dropout = params.get("network_dropout", 0.75)

                # Optimization Parameters
                batch_size = 1
                training_iters = self.n_iter
                epoch_size = 200

                Report += '\n\n---Savings---'
                Report += '\n Model saved in : '+ folder_model

                # Graph input
                x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
                y = tf.placeholder(tf.float32, shape=(batch_size*n_input, n_classes))
                keep_prob = tf.placeholder(tf.float32)

                # Call the model
                pred = Uconv_net(params, x)

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

                    stats = {'n_train': 0, 'n_train_pos': 0,
                             'n_test': 0, 'n_test_pos': 0,
                             'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'roc': 0.0,
                             'accuracy_history': [(0, 0)], 'precision_history': [(0, 0)], 'recall_history': [(0, 0)],
                             'roc_history': [(0, 0)],
                             't0': time.time(), 'total_fit_time': 0.0}

                    while step * batch_size < training_iters:
                        batch_x, batch_y = self.data_train.next_batch(batch_size, rnd = True, augmented_data= True)
                        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                                       keep_prob: dropout})

                        if step % display_step == 0:
                            # Calculate batch loss and accuracy
                            loss, acc, p = sess.run([cost, accuracy, pred], feed_dict={x: batch_x,
                                                                              y: batch_y,
                                                                              keep_prob: 1.})
                            if verbose == 2:
                                outputs = "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                                    "{:.5f}".format(acc)
                                print outputs

                        if step % epoch_size == 0 :

                            results_stats(self.data_test, sess, [stats['n_train'], trials.tids[-1]], stats)

                        if step % save_step == 0:
                            evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
                            with open(self.results_path+'/evolution.pkl', 'wb') as handle:
                                pickle.dump(evolution, handle)
                            save_path = saver.save(sess, self.results_path+"/model.ckpt")

                            print("Model saved in file: %s" % save_path)

                        step += 1

                stats['total_fit_time'] = time.time() - stats['t0']
                y_true, y_pred = results_stats(self.data_test, sess, [stats['n_train'], trials.tids[-1]], stats)

                score = param_hyperopt['fct'](y_true, y_pred)  # Score to maximize

                return {'loss': -score, 'status': STATUS_OK, 'eval_time': stats['total_fit_time']}

            # Trials object: results report
            trials = Trials()
            # Documentation: https://github.com/hyperopt/hyperopt/wiki/FMin
            best = fmin(hyperparam_train_test, model_hyperparam_hyperopt, algo=self.param_hyperopt['algo'],
                        max_evals=self.param_hyperopt['nb_eval'], trials=trials)
            print(best)
            # Save results
            pickle.dump(trials.trials, open(self.results_path + 'trials.pkl', "wb"))

        else:
            print ' '
            print 'Please provide a hyper parameter dict (called \'model_hyperparam\') in your classifier_model dict'
            print ' '

    def set_hyperopt_train(self, path_trainingset = '', path_best_trial='', save_trainable = True):
        ###############################################################################################################
        #
        # IF path_best_trial='':
        #       - Open all trials_*.pkl generated by hyperparam_optimization
        #       - Find better score: save related trials_*.pkl file as trials_best.pkl
        # ELSE:
        #       - load path_best_trial
        #
        # Set optimized params to self.model and save it as self.model_name + '_opt'
        #
        # Train the model
        #
        ###############################################################################################################

        if self.path_best_trial == '':
            fname_trials = [f for f in listdir(self.results_path) if
                            isfile(join(self.results_path, f)) and f.startswith('trials_')]

            trials_score_list = []
            trials_eval_time_list = []
            for f in fname_trials:
                with open(self.results_path + f) as outfile:
                    trial = pickle.load(outfile)
                    outfile.close()
                loss_list = [trial[i]['result']['loss'] for i in range(len(trial))]
                eval_time_list = [trial[i]['result']['eval_time'] for i in range(len(trial))]
                trials_score_list.append(min(loss_list))
                trials_eval_time_list.append(sum(eval_time_list))

            self.stats['total_hyperopt_time'] = sum(trials_eval_time_list)

            idx_best_trial = trials_score_list.index(min(trials_score_list))
            with open(self.results_path + fname_trials[idx_best_trial]) as outfile:
                best_trial = pickle.load(outfile)
                pickle.dump(best_trial, open(self.results_path + 'best_trial.pkl', "wb"))
                outfile.close()
        else:
            with open(path_best_trial) as outfile:
                best_trial = pickle.load(outfile)
                outfile.close()

        loss_list = [best_trial[i]['result']['loss'] for i in range(len(best_trial))]
        idx_best_params = loss_list.index(min(loss_list))
        best_params = best_trial[idx_best_params]['misc']['vals']

        if self.model_hyperparam is not None:

            model_hyperparam_opt = {}
            for k in self.model_hyperparam.keys():
                if isinstance(best_params[k][0], int):
                    model_hyperparam_opt[k] = self.model_hyperparam[k][best_params[k][0]]
                else:
                    model_hyperparam_opt[k] = float(best_params[k][0])

            # Network Parameters
            image_size = model_hyperparam_opt.get("network_image_size", 0.0005)
            n_input = image_size * image_size
            n_classes = model_hyperparam_opt.get("network_n_classes", 2)

            if path_trainingset == '':
                data_train = self.data_train
                data_test = self.data_test

            # Divers variables
            Loss = []
            Epoch = []
            Accuracy = []
            Report = ''
            verbose = 1

            # Results and Models
            folder_model = self.results_path+'/best_model'
            if not os.path.exists(folder_model):
                os.makedirs(folder_model)

            # --------------------SAME ALGORITHM IN TRAIN_model---------------------------

            # Network Parameters
            image_size = model_hyperparam_opt.get("network_image_size", 0.0005)
            n_input = image_size * image_size
            n_classes = model_hyperparam_opt.get("network_n_classes", 2)
            dropout = model_hyperparam_opt.get("network_dropout", 0.75)
            # Optimization Parameters
            batch_size = 1
            training_iters = self.n_iter
            epoch_size = 200
            display_step = 100
            save_step = 600

            Report += '\n\n---Savings---'
            Report += '\n Model saved in : ' + self.results_path+'/best_model'

            # Graph input
            x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
            y = tf.placeholder(tf.float32, shape=(batch_size * n_input, n_classes))
            keep_prob = tf.placeholder(tf.float32)

            # Call the model
            pred = Uconv_net(model_hyperparam_opt, x)

            # Define loss and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

            tf.scalar_summary('Loss', cost)

            temp = set(tf.all_variables())  # trick to get variables generated by the optimizer

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            init = tf.initialize_all_variables()

            if save_trainable:
                saver = tf.train.Saver(tf.trainable_variables())

            else:
                saver = tf.train.Saver(tf.all_variables())

            # Launch the graph
            Report += '\n\n---Intermediary results---\n'

            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                last_epoch = 0
                if path_model_init:
                    folder_restored_model = path_model_init
                    saver.restore(sess, folder_restored_model + "/model.ckpt")

                    if save_trainable:
                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

                    file = open(folder_restored_model + '/evolution.pkl', 'r')
                    evolution_restored = pickle.load(file)
                    last_epoch = evolution_restored["steps"][-1]

                else:
                    sess.run(init)
                print 'training start'

                step = 1
                epoch = 1 + last_epoch

                while step * batch_size < training_iters:
                    batch_x, batch_y = data_train.next_batch(batch_size, rnd=True, augmented_data=True)
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                                   keep_prob: dropout})

                    if step % display_step == 0:
                        # Calculate batch loss and accuracy
                        loss, acc, p = sess.run([cost, accuracy, pred], feed_dict={x: batch_x,
                                                                                   y: batch_y,
                                                                                   keep_prob: 1.})
                        if verbose == 2:
                            outputs = "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                                      "{:.5f}".format(acc)
                            print outputs

                    if step % epoch_size == 0:
                        start = time.time()
                        A = []  # list of accuracy scores on the datatest
                        L = []  # list of the Loss, or cost, scores on the dataset

                        data_test.set_batch_start()
                        for i in range(data_test.set_size):
                            batch_x, batch_y = data_test.next_batch(batch_size, rnd=False, augmented_data=False)
                            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

                            A.append(acc)
                            L.append(loss)

                            if verbose >= 1:
                                print '--\nAccuracy on patch' + str(i) + ': ' + str(acc)
                                print 'Loss on patch' + str(i) + ': ' + str(loss)
                        Accuracy.append(np.mean(A))
                        Loss.append(np.mean(L))
                        Epoch.append(epoch)

                        output_2 = '\n----\n Last epoch: ' + str(epoch)
                        output_2 += '\n Accuracy: ' + str(np.mean(A)) + ';'
                        output_2 += '\n Loss: ' + str(np.mean(L)) + ';'
                        print '\n\n----Scores on test:---' + output_2
                        epoch += 1

                    if step % save_step == 0:
                        evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
                        with open(folder_model + '/evolution.pkl', 'wb') as handle:
                            pickle.dump(evolution, handle)
                        save_path = saver.save(sess, folder_model + "/model.ckpt")

                        print("Model saved in file: %s" % save_path)
                        file = open(folder_model + "/report.txt", 'w')
                        file.write(Report + output_2)
                        file.close()

                    step += 1

                save_path = saver.save(sess, folder_model + "/model.ckpt")

                evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
                with open(folder_model + '/evolution.pkl', 'wb') as handle:
                    pickle.dump(evolution, handle)

                print("Model saved in file: %s" % save_path)
                print "Optimization Finished!"

        else:
            print ' '
            print 'Please provide a hyper parameter dict (called \'model_hyperparam\') in your classifier_model dict'
            print ' '


# To Call the training in the terminal

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path_training", required=True, help="")
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-m_init", "--path_model_init", required=False, help="")
    ap.add_argument("-lr", "--learning_rate", required=False, help="")
    ap.add_argument("-c", "--config_file", required=True,help="")
    ap.add_argument("-opt", "--path_param_training", required=True,help="")


    args = vars(ap.parse_args())
    path_training = args["path_training"]
    path_model = args["path_model"]
    path_model_init = args["path_model_init"]
    path_param_training = args["path_param_training"]
    config_file = args["config_file"]
    learning_rate = args["learning_rate"]
    if learning_rate :
        learning_rate = float(args["learning_rate"])
    else :
        learning_rate = None
        
    with open(config_file, 'r') as fd:
        config= json.loads(fd.read())

    # A REVOIR
    my_train = Trainer(path_training, path_model, path_param_training)
    my_train.hyperparam_optimization()
 