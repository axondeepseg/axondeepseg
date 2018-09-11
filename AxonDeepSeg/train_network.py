# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import pickle
from .data_management.input_data import input_data
from .config_tools import generate_config
from AxonDeepSeg.network_construction import *
from AxonDeepSeg.train_network_tools import *
from datetime import datetime
import time
import AxonDeepSeg.ads_utils


def train_model(path_trainingset, path_model, config, path_model_init=None,
                save_trainable=True, gpu=None, debug_mode=False, gpu_per = 1.0):
    """
    Main function. Trains a model using the configuration parameters.
    :param path_trainingset: Path to access the trainingset.
    :param path_model: Path indicating where to save the model.
    :param config: Dict, containing the configuration parameters of the network.
    :param path_model_init: Path to where the model to use for initialization is stored.
    :param save_trainable: Boolean. If True, only saves in the model variables that are trainable (evolve from gradient)
    :param gpu: String, name of the gpu to use. Prefer use of CUDA_VISIBLE_DEVICES environment variable.
    :param debug_mode: Boolean. If activated, saves more information about the distributions of
    most trainable variables, and also outputs more information.
    :param gpu_per: Float, between 0 and 1. Percentage of GPU to use.
    :return: Nothing.
    """ 
  
    ###################################################################################################################
    ############################################## VARIABLES INITIALIZATION ###########################################
    ###################################################################################################################
 
    # Results and Models
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Translating useful variables from the config file.
    learning_rate = config["learning_rate"]
    dropout = config["dropout"]
    weighted_cost = config["weighted_cost-activate"]
    batch_size_training = config["batch_size"]
    batch_size_validation = 8
    batch_norm_decay = config["batch_norm_decay_starting_decay"]

    image_size = config["trainingset_patchsize"]
    thresh_indices = config["thresholds"]
    n_classes = config["n_classes"]

    data_augmentation = generate_dict_da(config)
    weights_modifier = generate_dict_weights(config)
    data_aug_verbose = 0
   
    # Loading the datasets
    data_train = input_data(trainingset_path=path_trainingset, config=config, type_='train', batch_size=batch_size_training)
    data_validation = input_data(trainingset_path=path_trainingset, config=config, type_='validation', batch_size=batch_size_validation)
    
    n_iter_val = int(np.ceil(float(data_validation.set_size)/batch_size_validation))

    # Main loop parameters
    max_epoch = 2500
    epoch_size = data_train.epoch_size

    # Model saving frequency variable.
    if "save_epoch_freq" in config:
        save_last_epoch_freq = config["save_epoch_freq"]
    else:
        save_last_epoch_freq = 5
    save_best_moving_avg_epoch_freq = 5
    save_best_moving_avg_window = 10
    
    # Initializing the text to write in report.txt
    Report = ''
    output_2 = ''
    verbose = 1
    Report += '\n\n---Savings---'
    Report += '\n Model saved in : ' + path_model
    Report += '\n\n---PARAMETERS---\n'
    Report += 'learning_rate : ' + str(learning_rate) + '; \n batch_size :  ' + str(batch_size_training) + ';\n depth :  ' + str(
        config["depth"]) \
              + ';\n epoch_size: ' + str(epoch_size) + ';\n dropout :  ' + str(dropout) \
              + ';\n (if model restored) restored_model :' + str(path_model_init)

    ###################################################################################################################
    ############################################# GRAPH CONSTRUCTION ##################################################
    ###################################################################################################################
    
    ### ----------------------------------------------------------------------------------------------------------- ###
    #### 1 - Declaring the placeholders and other Tensorflow variables
    ### ----------------------------------------------------------------------------------------------------------- ###
    
    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size), name="input") # None relates to batch_size
    y = tf.placeholder(tf.float32, shape=(None, image_size, image_size, n_classes), name="ground_truth")
    phase = tf.placeholder(tf.bool, name="training_phase") # True if training phase, False for other phases.
    if weighted_cost == True:
        spatial_weights = tf.placeholder(tf.float32, shape=(None, image_size, image_size), name="spatial_weights") 
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    adapt_learning_rate = tf.placeholder(tf.float32, name="learning_rate") # If the learning rate changes over epochs
    adapt_bn_decay = tf.placeholder(tf.float32, name="batch_norm_decay") # If the learning rate changes over epochs
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Placeholders for metrics
    L_training_loss = tf.placeholder(tf.float32, name="List_training_loss")
    L_training_acc = tf.placeholder(tf.float32, name="List_training_acc")
    L_dice_myelin = tf.placeholder(tf.float32, name="List_dice_myelin")
    L_dice_axon = tf.placeholder(tf.float32, name="List_dice_axon")

    # Implementation note :
    # we could use a spatial_weights tensor with only ones, which would greatly simplify the rest of the code by
    # removing a lot of if conditions. Nevertheless, for computational reasons, we prefer to avoid the multiplication
    # by the spatial weights if the associated matrix is composed of only ones.
    # This position may be revised in the future.
  
    ### ----------------------------------------------------------------------------------------------------------- ###
    #### 2 - Decaying hyperparameters and processing the computation graph associated to the prediction
    ####     made by the U-net.
    ### ----------------------------------------------------------------------------------------------------------- ###

    # First, we decay the learning rate.
    # Note: if we use a polynomial decay, we also update the maximum number of epochs to be equal to the period of
    # the decay, since the learning rate will be equal to 0 after this period.

    if config["learning_rate_decay_activate"]:
        # Each decay period is expressed in number of images seen
        if config["learning_rate_decay_type"] == 'polynomial':
            adapt_learning_rate = poly_decay(global_step * batch_size_training, learning_rate,
                                             config["learning_rate_decay_period"])
            max_epoch = config["learning_rate_decay_period"] / epoch_size
        else:
            adapt_learning_rate = tf.train.exponential_decay(learning_rate, global_step * batch_size_training,
                                                             int(config["learning_rate_decay_period"]),
                                                             config["learning_rate_decay_rate"], staircase=False)
        tf.summary.scalar('adapt_lr', adapt_learning_rate)

    else:
        adapt_learning_rate = learning_rate

    # We also update the batch_norm_decay if needed
    if config["batch_norm_decay_decay_activate"]:
        adapt_bn_decay = inverted_exponential_decay(batch_norm_decay,
                                                    config["batch_norm_decay_ending_decay"],
                                                    global_step * batch_size_training,
                                                    config["batch_norm_decay_decay_period"], staircase=False)
        tf.summary.scalar('adapt_bnd', adapt_bn_decay)

    else:
        adapt_bn_decay = None
    
    # Next, we construct the computational graph linking the input and the prediction.
    pred = uconv_net(x, config, phase, bn_updated_decay = adapt_bn_decay)

    # We also display the total number of variables
    output_params = count_number_parameters(tf.trainable_variables())
    print(("Total number of parameters to train: " + str(output_params)))
    Report += '\n'+str(output_params)+'\n'
    
    ### ----------------------------------------------------------------------------------------------------------- ###
    #### 3 - Adapting the dimensions of the different tensors, then defining the optimization of the graph (loss + opt.)
    ### ----------------------------------------------------------------------------------------------------------- ###

    # Reshaping pred and y so that they are understandable by softmax_cross_entropy 
    with tf.name_scope('preds_reshaped'):
        pred_ = tf.reshape(pred, [-1,tf.shape(pred)[-1]])
    with tf.name_scope('y_reshaped'):    
        y_ = tf.reshape(y, [-1,tf.shape(y)[-1]])
   
    # Define loss and optimizer
    with tf.name_scope('cost'):
        if weighted_cost == True:    
            spatial_weights_ = tf.reshape(spatial_weights,[-1])
            cost = tf.reduce_mean(tf.multiply(spatial_weights_,
                                              tf.nn.softmax_cross_entropy_with_logits(logits=pred_, labels=y_)))
        else:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_, labels=y_))

    # We then define the Adam optimization operation.
    # We do it in two times to retrieve the gradients, so we can use them in TensorBoard.
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
        optimizer = tf.train.AdamOptimizer(learning_rate=adapt_learning_rate).apply_gradients(grads_list,
                                                                                              global_step=global_step)

    ### ----------------------------------------------------------------------------------------------------------- ###
    #### 4 - Constructing the metrics and storing the results to visualise them in TensorBoard
    ### ----------------------------------------------------------------------------------------------------------- ###
     
    # We evaluate the accuracy pixel-by-pixel
    correct_pred = tf.equal(tf.argmax(pred_, 1), tf.argmax(y_, 1))
    
    with tf.name_scope('accuracy_'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    # Now we create the pixel-wise dice
    pred_absolute = tf.cast(tf.equal(pred, tf.expand_dims(tf.reduce_max(pred, axis=-1),axis=-1)), tf.float32)
    
    dice = pw_dices(pred_absolute, tf.reshape(y,[-1, tf.shape(y)[1]*tf.shape(y)[2], tf.shape(y)[-1]]))
    dice = tf.reduce_mean(dice, axis=0)
    _, pw_dice_myelin, pw_dice_axon = tf.split(dice, n_classes, axis=0)
    
    # Defining list variables to keep track of the train error over one whole epoch
    # instead of just one batch (these are the ones we are going to summarize)
    training_loss = tf.reduce_mean(L_training_loss)
    training_acc = tf.reduce_mean(tf.cast(L_training_acc, tf.float32))
    dice_myelin = tf.reduce_mean(L_dice_myelin)
    dice_axon = tf.reduce_mean(L_dice_axon)

    ### ----------------------------------------------------------------------------------------------------------- ###
    #### 5 - Processing summaries (numerical and images) that are used to visualize the training phase metrics on TensorBoard
    ### ----------------------------------------------------------------------------------------------------------- ###

    tf.summary.scalar('loss', training_loss)
    tf.summary.scalar('accuracy', training_acc)
    tf.summary.scalar('dice_myelin', dice_myelin)
    tf.summary.scalar('dice_axon', dice_axon)

    # Debugging metric summaries
    if debug_mode == True:
        
        # Creation of a collection containing only the information we want to summarize
        for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if ('Adam' not in e.name) and (('weights' in e.name) or ('moving' in e.name) or ('bias' in e.name)):
                tf.add_to_collection('vals_to_summarize', e)
            
        summary_activations = tf.contrib.layers.summarize_collection("activations",
                                                                     summarizer=tf.contrib.layers.summarize_activation)
        summary_variables = tf.contrib.layers.summarize_collection('vals_to_summarize', name_filter=None)

    # We create a merged summary. It gathers all numeric data (histograms, kernels ... etc)
    merged_summaries = tf.summary.merge_all()

    # We only save the images if we are in debug mode.
    if debug_mode:

        # We also create a summary specific to images.
        # We add images of the input and the probability maps predicted by the u-net
        L_im_summ = []
        L_im_summ.append(tf.summary.image('input_image', tf.expand_dims(x, axis = -1)))
        L_im_summ.append(tf.summary.image('mask', y))
    
        # Creating the operation giving the probabilities
        with tf.name_scope('prob_maps'):
            # We compute the softmax predictions and reshape them to (b_s, imsz, imsz, n_classes)
            softmax_pred = tf.reshape(tf.reshape(tf.nn.softmax(pred_),
                                                 (-1, image_size * image_size, n_classes)),
                                      (-1, image_size, image_size, n_classes))
            probability_maps = tf.split(softmax_pred, n_classes, axis=3)
    
        # Adding a probability map for each class to the image summary
        for i, probmap in enumerate(probability_maps):
            L_im_summ.append(tf.summary.image('probability_map_class_'+str(i), probmap))
    
        # Merging the image summary
        images_merged_summaries = tf.summary.merge(L_im_summ)

    ### ----------------------------------------------------------------------------------------------------------- ###
    #### 6 - Initializing variables and summaries
    ### ----------------------------------------------------------------------------------------------------------- ###
 
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

    ###################################################################################################################
    ################################################ TRAINING PHASE ###################################################
    ###################################################################################################################

    Report += '\n\n---Intermediary results---\n'
    
    # Limiting the memory used by the training
    config_gpu = tf.ConfigProto(log_device_placement=True)
    config_gpu.gpu_options.per_process_gpu_memory_fraction = gpu_per
    #config_gpu.gpu_options.allow_growth = True # Activate if you only want to be dynamically attributed GPU memory.

    with tf.Session(config=config_gpu) as session:
        
        # Session initialized !
        
        ### ------------------------------------------------------------------------------------------------------- ###
        #### 1 - Preparing the main loop
        ### ------------------------------------------------------------------------------------------------------- ###

        # Initialization of useful variables
        last_epoch = 0
        Loss = []
        Epoch = []
        Accuracy = []
        epoch_training_loss = []
        epoch_training_acc = []
        epoch_validation_loss = []
        epoch_validation_acc = []
        epoch_training_dice_myelin = []
        epoch_training_dice_axon = []
        epoch_validation_dice_myelin = []
        epoch_validation_dice_axon = []
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
                session.run(tf.global_variables_initializer())

            file = open(folder_restored_model + '/evolution.pkl', 'r')
            evolution_restored = pickle.load(file)
            last_epoch = evolution_restored["steps"][-1]
        # Else, initializing the variables
        else:
            session.run(init)
        print('training start')

        # Display some information about weight selection
        if weighted_cost == True:
            print('Weighted cost selected')
        else:
            print('Default cost selected')

        # Update state variables (useful with transfert learning)
        step = 1
        epoch = 1 + last_epoch

        ### ------------------------------------------------------------------------------------------------------- ###
        #### 2 - Main loop: training the neural network
        ### ------------------------------------------------------------------------------------------------------- ###

        if debug_mode:
            t0 = 0

        while epoch < max_epoch:
            ### --------------------------------------------------------------------------------------------------- ###
            #### a) Optimizing the network with the training set. Keep track of the metric on TensorBoard.
            ### --------------------------------------------------------------------------------------------------- ###

            # We define the feed_dict parameter depending on whether we use weighted cost or not.
            if weighted_cost == True:
                # Extracting the batches
                batch_x, batch_y, weight = data_train.next_batch_WithWeights(augmented_data_=data_augmentation,
                                                                             weights_modifier=weights_modifier,
                                                                             each_sample_once=False,
                                                                             data_aug_verbose=data_aug_verbose)
                # Generating the arguments of the session.run call.
                feed_dict_train = {x: batch_x, y: batch_y, spatial_weights: weight, keep_prob: dropout, phase: True}

            else:
                # Extracting the batches
                batch_x, batch_y = data_train.next_batch(augmented_data_=data_augmentation, each_sample_once=False,
                                                                             data_aug_verbose=data_aug_verbose)

                # Generating the arguments of the session.run call.
                feed_dict_train = {x: batch_x, y: batch_y, keep_prob: dropout, phase: True}

            # Compute the gradients by running the optimizer for each batch, and retrieve the metrics.
            stepcost, stepacc, _, step_dice_myelin, step_dice_axon = session.run(
                [cost, accuracy, optimizer, pw_dice_myelin, pw_dice_axon], feed_dict=feed_dict_train)

            # Saving this step metrics
            epoch_training_loss.append(stepcost)
            epoch_training_acc.append(stepacc)
            epoch_training_dice_myelin.append(step_dice_myelin)
            epoch_training_dice_axon.append(step_dice_axon)
                
            # Printing some info
            if verbose>=2:
                print(('epoch_size:'+str(epoch_size)+'-global_step:'+str(global_step)))
                
            # If we just finished an epoch, we summarize the performance of the
            # net on the training set to see it in TensorBoard.
            if step*batch_size_training % epoch_size == 0:
                    
                # Writing the summary (metrics, + images if in debug mode).
                # We do two separates cases in order to only have one session.run call, what should be better for
                # computation times purposes.

                feed_dict_summary_train = {
                            x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_training_loss,
                            L_training_acc: epoch_training_acc, L_dice_myelin: epoch_training_dice_myelin,
                            L_dice_axon:epoch_training_dice_axon, spatial_weights: weight, phase:True
                        }
                if debug_mode:
                    summary, im_summary = session.run(
                        [merged_summaries, images_merged_summaries], feed_dict=feed_dict_summary_train)

                    train_writer.add_summary(summary, epoch)
                    train_writer.add_summary(im_summary, epoch)
                else:
                    summary = session.run(
                        merged_summaries, feed_dict=feed_dict_summary_train)

                    train_writer.add_summary(summary, epoch)

                # We reset the accumulating variables.
                epoch_training_loss = []
                epoch_training_acc = []
                epoch_training_dice_myelin = []
                epoch_training_dice_axon = []
  
            ### --------------------------------------------------------------------------------------------------- ###
            #### b) Evaluating the performance on the validation set. Keep track of it on TensorBoard.
            ### --------------------------------------------------------------------------------------------------- ###
            
            # At the end of every epoch we compute the performance of our network on the validation set and we
            # save the summaries to see them on TensorBoard
            if step*batch_size_training % epoch_size == 0:

                # Initialisation of the lists that will store the metrics.
                epoch_validation_loss = []
                epoch_validation_acc = []
                epoch_validation_dice_myelin = []
                epoch_validation_dice_axon = []

                if debug_mode:
                    t1 = time.time()
                    dt01 = t1 - t0 # time taken for the training phase
                
                if weighted_cost == True:

                    # We compute the metrics for each batch of the validation set
                    for i in range(n_iter_val):
                        if weighted_cost:
                            # Extracting the batch
                            batch_x, batch_y, weight = data_validation.next_batch_WithWeights(
                                augmented_data_={'type':'none'},
                                weights_modifier=weights_modifier,
                                each_sample_once=True
                            )
                            # Generating the feed_dict parameter
                            feed_dict_val = {
                                x: batch_x, y: batch_y, spatial_weights: weight, keep_prob: 1., phase: False}

                        else:
                            # Extracting the batches
                            batch_x, batch_y = data_validation.next_batch(
                                augmented_data_={'type': 'none'},
                                each_sample_once=True
                            )
                            # Generating the feed_dict parameter
                            feed_dict_val = {x: batch_x, y: batch_y, keep_prob: 1., phase:False}

                        # We compute the metrics but do not run the optimizer this time.
                        step_loss, step_acc, step_dice_myelin, step_dice_axon = session.run(
                            [cost, accuracy, pw_dice_myelin, pw_dice_axon], feed_dict=feed_dict_val)

                        # We have computed each validation batch but they must not be taken with the same weight
                        # when calculating the batch_size, which is why we added a factor
                        # in front of each step loss / step acc.
                        factor = float(batch_x.shape[0])/data_validation.set_size
                        epoch_validation_loss.append(factor*step_loss)
                        epoch_validation_acc.append(factor*step_acc)
                        epoch_validation_dice_myelin.append(factor*step_dice_myelin)
                        epoch_validation_dice_axon.append(factor*step_dice_axon)

                    # Once we have gone over each batch, we aggregate the metrics and we fed them to the summary writers
                    epoch_validation_loss = [np.sum(epoch_validation_loss)]
                    epoch_validation_acc = [np.sum(epoch_validation_acc)]
                    epoch_validation_dice_myelin = [np.sum(epoch_validation_dice_myelin)]
                    epoch_validation_dice_axon = [np.sum(epoch_validation_dice_axon)]
                    
                    # Writing the summary for this validation epoch. Metrics (+ images if in debug mode)
                    feed_dict_summary_val = {
                                x: batch_x, y: batch_y, keep_prob: dropout, L_training_loss: epoch_validation_loss,
                                L_training_acc: epoch_validation_acc, L_dice_myelin:epoch_validation_dice_myelin,
                                L_dice_axon:epoch_validation_dice_axon, spatial_weights: weight, phase:False
                            }

                    if debug_mode:
                        summary, im_summary_val = session.run(
                            [merged_summaries, images_merged_summaries], feed_dict=feed_dict_summary_val)
                        validation_writer.add_summary(summary, epoch)
                        validation_writer.add_summary(im_summary_val, epoch)
                    else:
                        summary = session.run(
                            merged_summaries, feed_dict=feed_dict_summary_val)
                        validation_writer.add_summary(summary, epoch)

                ### ----------------------------------------------------------------------------------------------- ###
                #### c) Displaying the metrics.
                ### ----------------------------------------------------------------------------------------------- ###

                # We display the metrics (evaluated on the validation set).
                acc = np.mean(epoch_validation_acc)
                loss = np.mean(epoch_validation_loss)
                
                Accuracy.append(acc)
                Loss.append(loss)
                Epoch.append(epoch)
                output_2 = '\n----\n Last epoch: ' + str(epoch)
                output_2 += '\n Accuracy: ' + str(acc) + ';'
                output_2 += '\n Loss: ' + str(loss) + ';'
                
                output_training = str(datetime.now()) + '-epoch:'+str(epoch)+'-loss:'+str(loss) + '-acc:'+str(acc)
                print(output_training)
                
                if debug_mode:
                    t2 = time.time()
                    dt12 = t2 - t1 # Time needed for the validation phase
                    t0 = time.time() # Début d'une boucle
                    print(('time analysis-training:+'+str(dt01)+'-validating:'+str(dt12)))


                ### ----------------------------------------------------------------------------------------------- ###
                #### d) Saving the model as a checkpoint, the metrics in a pickle file and update the file report.txt
                ### ----------------------------------------------------------------------------------------------- ###

                # Saving the model if it's the best one. We only do this check at the end of an epoch
                if epoch == 1: # First epoch is 1, not 0
                    acc_current_best = acc
                    loss_current_best = loss

                # At each check frequency defined we look if the moving average is better
                elif ((epoch >= max(save_best_moving_avg_epoch_freq, save_best_moving_avg_window)) and (epoch%save_best_moving_avg_epoch_freq == 0)):
                    acc_moving_avg = np.mean(Accuracy[-save_best_moving_avg_window:])
                    loss_moving_avg = np.mean(Loss[-save_best_moving_avg_window:])
                    if acc_moving_avg > acc_current_best:
                        
                        save_path = saver.save(session, path_model + "/best_acc_model.ckpt")
                        acc_current_best = acc_moving_avg
                        print(("Best accuracy model saved in file: %s" % save_path))
                        
                        # Saving the evolution in a pickle file
                        evolution = {'loss': np.mean(Loss[-10:]), 
                                     'steps': epoch, 
                                     'accuracy': np.mean(Accuracy[-10:])}
                        
                        with open(path_model + '/best_acc_stats.pkl', 'wb') as handle:
                            pickle.dump(evolution, handle)
                        
                    if loss_moving_avg < loss_current_best:
                        
                        save_path = saver.save(session, path_model + "/best_loss_model.ckpt")
                        loss_current_best = loss_moving_avg
                        print(("Best loss model saved in file: %s" % save_path))
                        
                        # Saving the evolution in a pickle file
                        evolution = {'loss': np.mean(Loss[-10:]), 
                                     'steps': epoch, 
                                     'accuracy': np.mean(Accuracy[-10:])}
                        
                        with open(path_model + '/best_loss_stats.pkl', 'wb') as handle:
                            pickle.dump(evolution, handle)

                # Moreover at a frequency we save the model regardless of the performance
                if epoch % save_last_epoch_freq == 0:

                    evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
                    with open(path_model + '/evolution.pkl', 'wb') as handle:
                        pickle.dump(evolution, handle)
                    save_path = saver.save(session, path_model + "/model.ckpt")

                    print(("Model saved in file: %s" % save_path))

                    evolution = {'loss': np.mean(Loss[-10:]), 
                                 'steps': epoch, 
                                 'accuracy': np.mean(Accuracy[-10:])}

                    with open(path_model + '/evolution_stats.pkl', 'wb') as handle:
                        pickle.dump(evolution, handle)
                    
                    file = open(path_model + "/report.txt", 'w')
                    file.write(Report + output_2)
                    file.close()
                
                # Increase the epoch #
                epoch += 1
                
            # Increase the step #
            step += 1

        # At the end of the training we save the model in a checkpoint file
        save_path = saver.save(session, path_model + "/model.ckpt")

        # Initialize best model with model after epoch 1
        evolution = {'loss': Loss, 'steps': Epoch, 'accuracy': Accuracy}
        with open(path_model + '/evolution.pkl', 'wb') as handle:
            pickle.dump(evolution, handle)

        print(("Model saved in file: %s" % save_path))
        print("Optimization Finished!")



def pw_dices(prediction, gt):
    """
    Computes the pixel-wise dice from the prediction tensor outputted by the network.
    :param prediction: Tensor, the prediction outputted by the network. Shape (N,H,W,C).
    :param gt: Tensor, the gold standard we work with. Shape (N,H,W,C).
    :return: Vector, dice per class for the current batch.
    """

    sum_ = tf.reduce_sum(prediction, axis=1) + tf.reduce_sum(gt, axis=1)
    intersection = tf.logical_and(tf.cast(prediction, tf.bool), tf.cast(gt, tf.bool))
    return tf.multiply(2., tf.div(tf.reduce_sum(tf.cast(intersection, tf.float32), axis=1), sum_))
    
        
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
