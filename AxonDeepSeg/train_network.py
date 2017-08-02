# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import math
import numpy as np
import os
import pickle
from data.input_data import input_data
from config_tools import generate_config
from AxonDeepSeg.train_network_tools import *


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

def train_model(path_trainingset, path_model, config, path_model_init=None,
                save_trainable=True, gpu=None, debug_mode=False, gpu_per = 1.0):
    """
    Principal function of this script. Trains the model using TensorFlow.
    
    :param path_model: path to save the trained model
    :param config: dict: network's parameters described in the header.
    :param path_model_init: (option) path of the model to initialize  the training
    :param save_trainable: if True, only weights are saved. If false the variables from the optimisers are saved too
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return:
    """
  
    ########################################################################################################################
    ############################################## VARIABLES INITIALIZATION ################################################
    ########################################################################################################################
 
    # Diverses variables
    Loss = []
    Epoch = []
    Accuracy = []
    Report = ''
    output_2 = ''
    verbose = 1
    
    # Results and Models
    folder_model = path_model
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)
 
    display_step = 100
    save_step = 600

    # Network Parameters

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
    batch_size = config["network_batch_size"]
    data_augmentation = config["network_data_augmentation"]
    batch_norm = config["network_batch_norm"]
    batch_norm_decay = config["network_batch_norm_decay"]
    image_size = config["network_trainingset_patchsize"]
    
    # Decay parameters
    additional_parameters = config["network_additional_parameters"]
    batch_norm_decay_decay_activate = additional_parameters["batch_norm_decay_decay_activate"]
    batch_norm_decay_ending_decay = additional_parameters["batch_norm_decay_ending_decay"]
    batch_norm_decay_decay_period = additional_parameters["batch_norm_decay_decay_period"]
    learning_rate_decay_activate = additional_parameters["learning_rate_decay_activate"]
    learning_rate_decay_period = additional_parameters["learning_rate_decay_period"]
    learning_rate_decay_rate = additional_parameters["learning_rate_decay_rate"]
    
    batch_size_validation = 8
    n_input = image_size * image_size


    # SAVING HYPERPARAMETERS TO USE THEM FOR apply_model. DEPRECATED, NOT USED -> TO DELETE

    hyperparameters = {'depth': depth, 'dropout': dropout, 'image_size': image_size,
                       'model_restored_path': path_model_init, 'learning_rate': learning_rate,
                       'network_n_classes': n_classes, 'network_downsampling': downsampling,
                       'network_thresholds': thresh_indices, 'weighted_cost': weighted_cost,
                       'network_convolution_per_layer': number_of_convolutions_per_layer,
                       'network_size_of_convolutions_per_layer': size_of_convolutions_per_layer,
                       'network_features_per_convolution': features_per_convolution,
                       'network_batch_size': batch_size}

    with open(folder_model + '/hyperparameters.pkl', 'wb') as handle:
        pickle.dump(hyperparameters, handle)

        
    # Loading the datasets
    data_train = input_data(trainingset_path=path_trainingset, type_='train', batch_size=batch_size,
                            thresh_indices=thresh_indices, image_size=image_size)
    data_validation = input_data(trainingset_path=path_trainingset, type_='validation', batch_size=batch_size_validation,
                                 thresh_indices=thresh_indices, image_size=image_size)
    
    n_iter_val = int(np.ceil(float(data_validation.set_size)/batch_size_validation))

    # Main loop parameters
    
    max_epoch = 2500
    epoch_size = data_train.epoch_size
    # batch_size is defined in the config file
    
    # Initilizating the text to write in report.txt
    Report += '\n\n---Savings---'
    Report += '\n Model saved in : ' + folder_model
    Report += '\n\n---PARAMETERS---\n'
    Report += 'learning_rate : ' + str(learning_rate) + '; \n batch_size :  ' + str(batch_size) + ';\n depth :  ' + str(
        depth) \
              + ';\n epoch_size: ' + str(epoch_size) + ';\n dropout :  ' + str(dropout) \
              + ';\n (if model restored) restored_model :' + str(path_model_init)

            
    ########################################################################################################################
    ################################################## GRAPH CONSTRUCTION ##################################################
    ########################################################################################################################
    
    ### ------------------------------------------------------------------------------------------------------------------ ###
    #### 1 - Declaring the placeholders and other variables
    ### ------------------------------------------------------------------------------------------------------------------ ###
    
    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size), name="input") # None relates to batch_size
    y = tf.placeholder(tf.float32, shape=(None, image_size, image_size, n_classes), name="ground_truth")
    phase = tf.placeholder(tf.bool, name="training_phase") # Tells us if we are in training phase of test phase. Used for batch_normalization
    if weighted_cost == True:
        spatial_weights = tf.placeholder(tf.float32, shape=(None, image_size, image_size), name="spatial_weights") 
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    adapt_learning_rate = tf.placeholder(tf.float32, name="learning_rate") # If the learning rate changes over epochs
    adapt_bn_decay = tf.placeholder(tf.float32, name="batch_norm_decay") # If the learning rate changes over epochs
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Implementation note : we could use a spatial_weights tensor with only ones, which would greatly simplify the rest of the code by removing a lot of if conditions. Nevertheless, for computational reasons, we prefer to avoid the multipliciation by the spatial weights if the associated matrix is composed of only ones. This position may be revised in the future.
  
    ### ------------------------------------------------------------------------------------------------------------------ ###
    #### 2 - Creating the graph associated to the prediction made by the U-net.
    ### ------------------------------------------------------------------------------------------------------------------ ###
    
    # We update the batch_norm_decay if needed
    
    if batch_norm_decay_decay_activate:
        adapt_bn_decay = inverted_exponential_decay(batch_norm_decay, batch_norm_decay_ending_decay, global_step,
                                                    batch_norm_decay_decay_period*epoch_size, staircase=False)
        tf.summary.scalar('adapt_bnd', adapt_bn_decay)
    else:
        adapt_bn_decay = None
    
    # We select a GPU before creating the prediction graph. WARNING : THIS IS FOR BIRELI, THERE ARE ONLY 2 GPUs

    if gpu in ['gpu:0', 'gpu:1']:
        with tf.device('/' + gpu):
            pred = uconv_net(x, config, phase, bn_updated_decay = adapt_bn_decay)
    else:
        pred = uconv_net(x, config, phase, bn_updated_decay = adapt_bn_decay)

    # We also display the total number of variables
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes

    output_params = 'tot_param = ' + str(total_parameters)
    print(output_params)
    
    Report += '\n'+output_params+'\n'
    
    ### ------------------------------------------------------------------------------------------------------------------ ###
    #### 3 - Adapting the dimensions of the differents tensors, then defining the optimization of the graph (loss + opt.)
    ### ------------------------------------------------------------------------------------------------------------------ ###
    
    # First, we prepare the terrain for the decaying learning rate and we update (decay) the batch norm decay (which should be called batch norm momentum).
    if learning_rate_decay_activate:
        adapt_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 
                                                     int(learning_rate_decay_period*epoch_size/batch_size), learning_rate_decay_rate, staircase=True)
        tf.summary.scalar('adapt_lr', adapt_learning_rate)

    else:
        adapt_learning_rate = learning_rate
    

    # Reshaping pred and y so that they are understandable by softmax_cross_entropy 
    with tf.name_scope('preds_reshaped'):
        pred_ = tf.reshape(pred, [-1,tf.shape(pred)[-1]])
    with tf.name_scope('y_reshaped'):    
        y_ = tf.reshape(y, [-1,tf.shape(y)[-1]])
   
    # Define loss and optimizer
    with tf.name_scope('cost'):
        if weighted_cost == True:    
            spatial_weights_ = tf.reshape(spatial_weights,[-1])
            cost = tf.reduce_mean(tf.multiply(spatial_weights_,tf.nn.softmax_cross_entropy_with_logits(logits=pred_, labels=y_)))
        else:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_, labels=y_))

    #temp = set(tf.global_variables())  # trick to get variables generated by the optimizer
    
    # We then define the Adam optimization operation. We do it in two times to obtain the gradients, so we can use them in TensorBoard.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops (including the BN parameters) before performing the train_step
        # First we compute the gradients
        grads_list = tf.train.AdamOptimizer(learning_rate=adapt_learning_rate).compute_gradients(cost)
        
        if debug_mode == True:
            # We make a summary of the gradients
            for grad, weight in grads_list:
                if 'weight' in weight.name:
                    # here we can split weight name by ':' to avoid the warning message we're getting
                    weight_grads_summary = tf.summary.histogram('_'.join(weight.name.split(':')) + '_grad', grad)

        # Then we continue the optimization as usual
        optimizer = tf.train.AdamOptimizer(learning_rate=adapt_learning_rate).apply_gradients(grads_list, global_step=global_step)   
    

    ### ------------------------------------------------------------------------------------------------------------------ ###
    #### 4 - Evaluating the model and storing the results to visualise them in TensorBoard
    ### ------------------------------------------------------------------------------------------------------------------ ###
     
    # We evaluate the accuracy pixel-by-pixel
    correct_pred = tf.equal(tf.argmax(pred_, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy_'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Defining list variables to keep track of the train error over one whole epoch instead of just one batch (these are the ones we are going to summarize)
    
    L_training_loss = tf.placeholder(tf.float32, name="List_training_loss")
    L_training_acc = tf.placeholder(tf.float32, name="List_training_acc")

    training_loss = tf.reduce_mean(L_training_loss)
    training_acc = tf.reduce_mean(tf.cast(L_training_acc, tf.float32))
    
    tf.summary.scalar('loss', training_loss)
    tf.summary.scalar('accuracy', training_acc)
        
    # Creation of a collection containing only the information we want to summarize

    for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if ('Adam' not in e.name) and (('weights' in e.name) or ('moving' in e.name) or ('bias' in e.name)):
            tf.add_to_collection('vals_to_summarize', e)
            
    # Summaries   
    if debug_mode == True:
        summary_activations = tf.contrib.layers.summarize_collection("activations",
                                                                     summarizer=tf.contrib.layers.summarize_activation)
        summary_variables = tf.contrib.layers.summarize_collection('vals_to_summarize', name_filter=None)    

    
    ### ------------------------------------------------------------------------------------------------------------------ ###
    #### 5 - Processing summaries (numerical and images) that are used to visualize the training phase metrics on TensorBoard
    ### ------------------------------------------------------------------------------------------------------------------ ###
 
    # We create a merged summary. It relates to all numeric data (histograms, kernels ... etc)
    merged_summaries = tf.summary.merge_all()

    # We also create a summary specific to images. We add images of the input and the probability maps predicted by the u-net
    L_im_summ = []
    L_im_summ.append(tf.summary.image('input_image', tf.expand_dims(x, axis = -1)))
    if debug_mode:
        L_im_summ.append(tf.summary.image('mask', y))
    
    # Creating the operation giving the probabilities
    with tf.name_scope('prob_maps'):
        softmax_pred = tf.reshape(tf.reshape(tf.nn.softmax(pred_), (-1, image_size * image_size, n_classes)), (-1, image_size, image_size, n_classes)) # We compute the softmax predictions and reshape them to (b_s, imsz, imsz, n_classes)
        probability_maps = tf.split(softmax_pred, n_classes, axis=3)
    
    # Adding a probability map for each class to the image summary
    for i, probmap in enumerate(probability_maps):
        L_im_summ.append(tf.summary.image('probability_map_class_'+str(i), probmap))
    
    # Merging the image summary
    images_merged_summaries = tf.summary.merge(L_im_summ)

    ### ------------------------------------------------------------------------------------------------------------------ ###
    #### 6 - Initializing variables and summaries
    ### ------------------------------------------------------------------------------------------------------------------ ###
 
    # We create the directories where we will store our model
    train_writer = tf.summary.FileWriter(logdir=path_model + '/train')
    validation_writer = tf.summary.FileWriter(logdir=path_model + '/validation')

    # Initializing all the variables
    init = tf.global_variables_initializer()

    # Creating a tool to preserve the state of the variables (useful for transfer learning for instance)
    
    if save_trainable:
        #saver = tf.train.Saver(tf.trainable_variables(), tf.model_variables())
        saver = tf.train.Saver(tf.model_variables())
    else:
        saver = tf.train.Saver(tf.all_variables())

    ########################################################################################################################
    #################################################### TRAINING PHASE ####################################################
    ########################################################################################################################

    Report += '\n\n---Intermediary results---\n'
    
    # Limiting the memory used by the training
    config_gpu = tf.ConfigProto(log_device_placement=True)
    config_gpu.gpu_options.per_process_gpu_memory_fraction = gpu_per
    #config_gpu.gpu_options.allow_growth = True

    with tf.Session(config=config_gpu) as session:
        
        # Session initialized !
        
        ### --------------------------------------------------------------------------------------------------------------- ###
        #### 1 - Preparing the main loop
        ### --------------------------------------------------------------------------------------------------------------- ###

        # Initialization of useful variables
        last_epoch = 0
        epoch_training_loss = []
        epoch_training_acc = []
        epoch_validation_loss = []
        epoch_validation_acc = []
        acc_current_best = 0
        loss_current_best = 10000

        # Setting the graph in the summaries writer in order to be able to use TensorBoard
        train_writer.add_graph(session.graph)
        validation_writer.add_graph(session.graph)

        # Loading a previous session if requested.
        if path_model_init: 
            folder_restored_model = path_model_init
            saver.restore(session, folder_restored_model + "/model.ckpt")
            if save_trainable:
                session.run(tf.global_variables_initializer(set(tf.global_variables()) - temp))
            file = open(folder_restored_model + '/evolution.pkl', 'r')
            evolution_restored = pickle.load(file)
            last_epoch = evolution_restored["steps"][-1]
        # Else, initializing the variables
        else:
            session.run(init)
        print 'training start'

        # Display some information about weight selection
        if weighted_cost == True:
            print('Weighted cost selected')
        else:
            print('Default cost selected')

        # Update state variables (useful with transfert learning)
        step = 1
        epoch = 1 + last_epoch

        ### --------------------------------------------------------------------------------------------------------------- ###
        #### 2 - Main loop: training the neural network
        ### --------------------------------------------------------------------------------------------------------------- ###
        
        while epoch < max_epoch:
            
            ### ----------------------------------------------------------------------------------------------------------- ###
            #### a) Optimizing the network with the training set. Keep track of the metric on TensorBoard
            ### ----------------------------------------------------------------------------------------------------------- ###
            
            # Compute the optimizer at each training iteration
            if weighted_cost == True:
                # Extracting the batches
                batch_x, batch_y, weight = data_train.next_batch_WithWeights(augmented_data=data_augmentation,                                                                              weights_modifier=config["network_weighted_cost_parameters"],
                                                                             each_sample_once=False)
                  
                # Running the optimizer and computing the cost and accuracy.
                stepcost, stepacc, _ = session.run([cost, accuracy, optimizer], feed_dict={x: batch_x, y: batch_y,
                                                       spatial_weights: weight,
                                                       keep_prob: dropout, phase:True})
                epoch_training_loss.append(stepcost)
                epoch_training_acc.append(stepacc)
                
                # If we just finished an epoch, we summarize the performance of the
                # net on the training set to see it in TensorBoard.
                if step*batch_size % epoch_size == 0:            
                    # Writing the summary
                    summary, im_summary = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_training_loss, L_training_acc: epoch_training_acc, spatial_weights: weight, phase:True})
                    train_writer.add_summary(summary, epoch)
                    train_writer.add_summary(im_summary, epoch)
                    
                    epoch_training_loss = []
                    epoch_training_acc = []
  
            else: # No weighted cost
                # Extracting batches
                batch_x, batch_y = data_train.next_batch(augmented_data=data_augmentation, each_sample_once=False)
                
                # Computing loss, accuracy and optimizing the weights
                stepcost, stepacc, _ = session.run([cost, accuracy, optimizer], feed_dict={x: batch_x, y: batch_y,
                                                   keep_prob: dropout, phase:True})
                # Evaluating the loss and the accuracy for the dataset
                epoch_training_loss.append(stepcost)
                epoch_training_acc.append(stepacc)

                # If were just finished an epoch, we summarize the performance of the
                # net on the training set to see it in tensorboard.
                if step*batch_size % epoch_size == 0:
                                        
                    # Writing the summary
                    summary, im_summary = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_training_loss, L_training_acc: epoch_training_acc, phase:True})
                    
                    train_writer.add_summary(summary, epoch)
                    train_writer.add_summary(im_summary, epoch)

                    epoch_training_loss = []
                    epoch_training_acc = []

            ### ----------------------------------------------------------------------------------------------------------- ###
            #### b) Evaluating and displaying the performance on the training set
            ### ----------------------------------------------------------------------------------------------------------- ###

                    
            # Every now and then we display the performance of the network on the training set, on the current batch.
            # Note : this part is not really used right now.
            if step*batch_size % display_step == 0:
                # Calculate batch loss and accuracy
                if weighted_cost == True:
                    loss, acc, p = session.run([cost, accuracy, pred], feed_dict={x: batch_x, y: batch_y,
                                                                               spatial_weights: weight, keep_prob: 1., phase:False})
                else:
                    loss, acc, p = session.run([cost, accuracy, pred], feed_dict={x: batch_x, y: batch_y,
                                                                               keep_prob: 1., phase:False})
                if verbose == 2:
                    outputs = "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc)
                    print outputs

            ### ----------------------------------------------------------------------------------------------------------- ###
            #### c) Evaluating the performance on the validation set. Keep track of it on TensorBoard and in a pickle file.
            ### ----------------------------------------------------------------------------------------------------------- ###

            # At the end of every epoch we compute the performance of our network on the validation set and we
            # save the summaries to see them on TensorBoard
            if step*batch_size % epoch_size == 0:
                
                if weighted_cost == True:
                    epoch_validation_loss = []
                    epoch_validation_acc = []
                    for i in range(n_iter_val):

                        batch_x, batch_y, weight = data_validation.next_batch_WithWeights(augmented_data={'type':'none'},                                                                              weights_modifier=config["network_weighted_cost_parameters"],
                                                                                          each_sample_once=True)

                        step_loss, step_acc = session.run([cost, accuracy],
                                                feed_dict={x: batch_x, y: batch_y, spatial_weights: weight, keep_prob: 1., phase:False})
                        factor = float(batch_x.shape[0])/data_validation.set_size
                        epoch_validation_loss.append(factor*step_loss)
                        epoch_validation_acc.append(factor*step_acc)
                        
                    # We have computed each validation batch but they must not be taken with the same weight when calculating the batch_size, which is why we added a factor in front of each step loss / step acc.
                    epoch_validation_loss = [np.sum(epoch_validation_loss)]
                    epoch_validation_acc = [np.sum(epoch_validation_acc)]                    
                    
                    # Writing the summary for this step of the training, to use in Tensorflow
                    summary, im_summary_val = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_validation_loss, L_training_acc: epoch_validation_acc, spatial_weights: weight, phase:False})


                else:
                    epoch_validation_loss = []
                    epoch_validation_acc = []
                    for i in range(n_iter_val):    
                        batch_x, batch_y = data_validation.next_batch(augmented_data={'type':'none'}, each_sample_once=True)
                        step_loss, step_acc = session.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase:False})
                        
                        factor = float(batch_x.shape[0])/data_validation.set_size
                        epoch_validation_loss.append(factor*step_loss)
                        epoch_validation_acc.append(factor*step_acc)
                       

                    # We have computed each validation batch but they must not be taken with the same weight when calculating the batch_size, which is why we added a factor in front of each step loss / step acc.
                    epoch_validation_loss = [np.sum(epoch_validation_loss)]
                    epoch_validation_acc = [np.sum(epoch_validation_acc)]     
                                        
                    # Writing the summary for this step of the training, to use in Tensorflow
                    summary, im_summary_val = session.run([merged_summaries, images_merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_validation_loss, L_training_acc: epoch_validation_acc, phase:False})
                    
                validation_writer.add_summary(summary, epoch)
                validation_writer.add_summary(im_summary_val, epoch)

                # We also keep the metrics in lists so that we can save them in a pickle file later.
                # We display the metrics (evaluated on the validation set).
                
                acc = np.mean(epoch_validation_acc)
                loss = np.mean(epoch_validation_loss)
                
                Accuracy.append(acc)
                Loss.append(loss)
                Epoch.append(epoch)

                output_2 = '\n----\n Last epoch: ' + str(epoch)
                output_2 += '\n Accuracy: ' + str(acc) + ';'
                output_2 += '\n Loss: ' + str(loss) + ';'
                print '\n\n----Scores on validation:---' + output_2

                # Saving the model if it's the best one
                #if epoch == 1:
                #    acc_current_best = acc
                #    loss_current_best = loss

                    # If new model is better than the last one, update best model
                #elif (acc > acc_current_best and loss < loss_current_best):
                #    save_path = saver.save(session, folder_model + "/best_model.ckpt")

                epoch += 1

            ### ----------------------------------------------------------------------------------------------------------- ###
            #### d) Saving the model as a checkpoint, the metrics in a pickle file and update the file report.txt
            ### ----------------------------------------------------------------------------------------------------------- ###

            if step*batch_size % save_step == 0:
                evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
                with open(folder_model + '/evolution.pkl', 'wb') as handle:
                    pickle.dump(evolution, handle)
                save_path = saver.save(session, folder_model + "/model.ckpt")

                print("Model saved in file: %s" % save_path)
                file = open(folder_model + "/report.txt", 'w')
                file.write(Report + output_2)
                file.close()

            step += 1
    
        # At the end of each epoch we save the model in a checkpoint file
        save_path = saver.save(session, folder_model + "/model.ckpt")

        # Initialize best model with model after epoch 1
        evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
        with open(folder_model + '/evolution.pkl', 'wb') as handle:
            pickle.dump(evolution, handle)

        print("Model saved in file: %s" % save_path)
        print "Optimization Finished!"
        

def inverted_exponential_decay(a, b, global_step, decay_period, staircase=False):
    if staircase:
        return a + (b - a)*(1 - tf.exp(-tf.cast(global_step, tf.int32)/decay_period))
    else:
        return a + (b - a)*(1 - tf.exp(-tf.cast(global_step, tf.float64)/decay_period))


        
# To Call the training in the terminal

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path_training", required=True, help="")
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-co", "--config_file", required=False, help="", default="~/.axondeepseg.json")
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
