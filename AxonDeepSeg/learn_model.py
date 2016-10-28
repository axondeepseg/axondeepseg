import tensorflow as tf
import math
import numpy as np
import os
import pickle
import time
from learning.input_data import input_data
import sys

def learn_model(path_trainingset, path_model, path_model_init = None, learning_rate = None, save_trainable = True, verbose = 1):
    """
    :param path_trainingset: path of the train and test set built from data_construction
    :param path_model:
    :param path_model_init:
    :param learning_rate: learning_rate of the optimiser
    :param save_trainable: if True, only weights are saved. If false the variables from the optimisers are saved too
    :param verbose:
    :return:
    """

    if not learning_rate :
        learning_rate = 0.0005

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
    n_classes = 2
    dropout = 0.75
    depth = 6

    hyperparameters = {'depth': depth,'dropout': dropout, 'image_size': image_size,
                       'model_restored_path': path_model_init, 'learning_rate': learning_rate}

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
    def conv_net(x, weights, biases, dropout, image_size = image_size):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, image_size, image_size, 1])
        data_temp = x
        data_temp_size = [image_size]
        relu_results = []

    # contraction
        for i in range(depth):
          conv1 = conv2d(data_temp, weights['wc1'][i], biases['bc1'][i])
          conv2 = conv2d(conv1, weights['wc2'][i], biases['bc2'][i])
          relu_results.append(conv2)

          conv2 = maxpool2d(conv2, k=2)
          data_temp_size.append(data_temp_size[-1]/2)
          data_temp = conv2

        conv1 = conv2d(data_temp, weights['wb1'], biases['bb1'])
        conv2 = conv2d(conv1, weights['wb2'], biases['bb2'])
        data_temp_size.append(data_temp_size[-1])
        data_temp = conv2


    # expansion
        for i in range(depth):
            data_temp = tf.image.resize_images(data_temp, data_temp_size[-1] * 2, data_temp_size[-1] * 2)
            upconv = conv2d(data_temp, weights['upconv'][i], biases['upconv'][i])
            data_temp_size.append(data_temp_size[-1]*2)

    # concatenation
            upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[depth-i-1], [0, 0, 0, 0],
                                                                     [-1, data_temp_size[depth-i-1], data_temp_size[depth-i-1], -1]), upconv])
            conv1 = conv2d(upconv_concat, weights['we1'][i], biases['be1'][i])
            conv2 = conv2d(conv1, weights['we2'][i], biases['be2'][i])
            data_temp = conv2

    # final convolution and segmentation
        finalconv = tf.nn.conv2d(conv2, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
        final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], 2]))

        return final_result

    weights = {'wc1':[],'wc2':[],'we1':[],'we2':[],'upconv':[],'finalconv':[],'wb1':[], 'wb2':[]}
    biases = {'bc1':[],'bc2':[],'be1':[],'be2':[],'finalconv_b':[],'bb1':[], 'bb2':[],'upconv':[]}

    # Contraction
    for i in range(depth):
      if i == 0:
        num_features_init = 1
        num_features = 64
      else:
        num_features = num_features_init * 2


    # Store layers weight & bias

      weights['wc1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))), name = 'wc1-%s'%i))
      weights['wc2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name = 'wc2-%s'%i))
      biases['bc1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='bc1-%s'%i))
      biases['bc2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='bc2-%s'%i))

      image_size = image_size/2
      num_features_init = num_features
      num_features = num_features_init*2

    weights['wb1']= tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))),name='wb1-%s'%i)
    weights['wb2']= tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='wb2-%s'%i)
    biases['bb1']= tf.Variable(tf.random_normal([num_features]), name='bb2-%s'%i)
    biases['bb2']= tf.Variable(tf.random_normal([num_features]), name='bb2-%s'%i)

    num_features_init = num_features

    for i in range(depth):

        num_features = num_features_init/2
        weights['upconv'].append(tf.Variable(tf.random_normal([2, 2, num_features_init, num_features]), name='upconv-%s'%i))
        biases['upconv'].append(tf.Variable(tf.random_normal([num_features]), name='bupconv-%s'%i))
        weights['we1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))), name='we1-%s'%i))
        weights['we2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='we2-%s'%i))
        biases['be1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='be1-%s'%i))
        biases['be2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='be2-%s'%i))

        num_features_init = num_features

    weights['finalconv']= tf.Variable(tf.random_normal([1, 1, num_features, n_classes]), name='finalconv-%s'%i)
    biases['finalconv_b']= tf.Variable(tf.random_normal([n_classes]), name='bfinalconv-%s'%i)

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

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
        if path_model_init :
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
            batch_x, batch_y = data_train.next_batch(batch_size, rnd = True, augmented_data= True)
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
                start = time.time()
                A = []
                L = []

                data_test.set_batch_start()
                for i in range(data_test.set_size):
                    batch_x, batch_y = data_test.next_batch(batch_size, rnd=False, augmented_data= False)
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    A.append(acc)
                    L.append(loss)

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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path_training", required=True, help="")
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-m_init", "--path_model_init", required=False, help="")
    ap.add_argument("-lr", "--learning_rate", required=False, help="")

    args = vars(ap.parse_args())
    path_training = args["path_training"]
    path_model = args["path_model"]
    path_model_init = args["path_model_init"]
    learning_rate = args["learning_rate"]
    if learning_rate :
        learning_rate = float(args["learning_rate"])

    else : learning_rate = None

    learn_model(path_training, path_model, path_model_init, learning_rate)