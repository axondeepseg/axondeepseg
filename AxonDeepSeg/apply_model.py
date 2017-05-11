# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import os
import pickle
import numpy as np
from scipy import io
from scipy.misc import imread, imsave
from skimage.transform import rescale
from skimage import exposure
from config import general_pixel_size, path_matlab, path_axonseg

#import matplotlib.pyplot as plt 


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
        positions = [[0, 0]]
        patches = [patch]

    else:
        q_h, r_h = divmod(h, size)
        q_w, r_w = divmod(w, size)

        r2_h = size - r_h
        r2_w = size - r_w
        q2_h = q_h + 1
        q2_w = q_w + 1

        q3_h, r3_h = divmod(r2_h, q_h)
        q3_w, r3_w = divmod(r2_w, q_w)

        dataset = []
        positions = []
        pos = 0
        while pos + size <= h:
            pos2 = 0
            while pos2 + size <= w:
                patch = img[pos:pos + size, pos2:pos2 + size]
                patch = exposure.equalize_hist(patch)
                patch = (patch - np.mean(patch)) / np.std(patch)

                dataset.append(patch)
                positions.append([pos, pos2])
                pos2 = size + pos2 - q3_w
                if pos2 + size > w:
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
        reshaped_pred = np.reshape(pred, [256, 256])
        image[pos[0]:pos[0] + 256, pos[1]:pos[1] + 256] = reshaped_pred
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

    # Network Parameters
    image_size = image_size
    n_classes = config["network_n_classes"]
    depth = config["network_depth"]
    number_of_convolutions_per_layer = config["network_convolution_per_layer"]
    size_of_convolutions_per_layer = config["network_size_of_convolutions_per_layer"]
    features_per_convolution = config["network_features_per_convolution"]
    downsampling = config["network_downsampling"]

    # Reshape input picture
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
            print('Size:', size_of_convolutions_per_layer[i][conv_number])

            if conv_number == 0:
                convolution_e = conv2d(upconv_concat, weights['we'][i][conv_number], biases['be'][i][conv_number])
            else:
                convolution_e = conv2d(convolution_e, weights['we'][i][conv_number], biases['be'][i][conv_number])

        data_temp = convolution_e

    # final convolution and segmentation
    finalconv = tf.nn.conv2d(convolution_e, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
    final_result = tf.reshape(finalconv, tf.TensorShape(
        [finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], n_classes]))

    return final_result


def apply_convnet(path_my_data, path_model, config):
    """
    :param path_my_data: folder of the image to segment. Must contain image.png
    :param path_model: folder of the model of segmentation. Must contain model.ckpt
    :param config: dict: network's parameters described in the header.
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: prediction, the mask of the segmentation
    """

    print '\n\n ---Start axon segmentation on %s---' % path_my_data

    path_img = path_my_data + '/image.png'
    img = imread(path_img, flatten=False, mode='L')

    file = open(path_my_data + '/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())

    # set the resolution to the general_pixel_size
    rescale_coeff = pixel_size / general_pixel_size
    img = (rescale(img, rescale_coeff) * 256).astype(int)

    batch_size = 1

    ###############
    # Network Parameters
    image_size = 256
    thresh_indices = config["network_thresholds"]
    n_classes = config["network_n_classes"]
    depth = config["network_depth"]
    number_of_convolutions_per_layer = config["network_convolution_per_layer"]
    size_of_convolutions_per_layer = config["network_size_of_convolutions_per_layer"]
    features_per_convolution = config["network_features_per_convolution"]
    downsampling = config["network_downsampling"]
    ##############

    folder_model = path_model
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    if os.path.exists(folder_model + '/hyperparameters.pkl'):
        print 'hyperparameters detected in the model'
        hyperparameters = pickle.load(open(folder_model + '/hyperparameters.pkl', "rb"))
        image_size = hyperparameters['image_size']

    # --------------------SAME ALGORITHM IN TRAIN_model---------------------------

    x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))

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
    ####################################################

    # Call the model
    pred = uconv_net(x, config, weights, biases)

    saver = tf.train.Saver(tf.all_variables())

    # Image to batch
    image_init, data, positions = im2patches(img, 256)
    predictions = []

    # Launch the graph
    sess = tf.Session()
    saver.restore(sess, folder_model + '/model.ckpt')

    # --------------------- Apply the segmentation to each patch of the images--------------------------------

    for i in range(len(data)):
        print 'processing patch %s on %s' % (i + 1, len(data))
        batch_x = np.asarray([data[i]])
        p = sess.run(pred, feed_dict={x: batch_x})

        Mask = np.zeros_like(p[:, 0])
        for pixel in range(len(p[:, 0])):
            Mask[pixel] = np.argmax(p[pixel, :])
            #print(p[pixel, :])


        if np.max(Mask)!=0:
            Mask = Mask.reshape(256, 256)/np.max(Mask)

        else: 
            Mask = Mask.reshape(256,256)

        predictions.append(Mask)

    sess.close()
    tf.reset_default_graph()

    # -----------------------Merge each segmented patch to reconstruct the total segmentation

    h_size, w_size = image_init.shape
    prediction_rescaled = patches2im(predictions, positions, h_size, w_size)

    prediction = rescale(prediction_rescaled, 1 / rescale_coeff)

    # Rescaling and set indices to integer values
    for indice,value in enumerate(thresh_indices[:-1]):
        if np.max(prediction) > 1.001:
            thresh_inf = np.int(255*value)
            thresh_sup = np.int(255*thresh_indices[indice+1])
        else:
            thresh_inf = value
            thresh_sup = thresh_indices[indice+1]   

        prediction[(prediction >= thresh_inf) & (prediction < thresh_sup)] = np.mean([value,thresh_indices[indice+1]])

    prediction[(prediction >= thresh_indices[-1])] = 1

    return prediction

    #######################################################################################################################

def axon_segmentation(path_my_data, path_model, config):
    """
    :param path_my_data: folder of the image to segment. Must contain image.jpg
    :param path_model: folder of the model of segmentation. Must contain model.ckpt
    :return: no return
    Results including the prediction and the prediction associated with the mrf are saved in the image_path
    AxonMask.mat is saved in the image_path to feed the detection of Myelin
    /AxonSeg.jpeg is saved in the image_path
    """

    file = open(path_my_data + '/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())
    rescale_coeff = pixel_size / general_pixel_size

    # ------ Apply ConvNets ------- #
    prediction = apply_convnet(path_my_data, path_model, config)
    thresh_indices = config.get("network_thresholds", [0, 0.5])

    prediction = rescale(prediction.astype(float), rescale_coeff)

    for indice,value in enumerate(thresh_indices[:-1]):
        if np.max(prediction) > 1.001:
            thresh_inf = np.int(255*value)
            thresh_sup = np.int(255*thresh_indices[indice+1])
        else:
            thresh_inf = value
            thresh_sup = thresh_indices[indice+1]   

        prediction[(prediction >= thresh_inf) & (prediction < thresh_sup)] = np.mean([value,thresh_indices[indice+1]])

    prediction[(prediction >= thresh_indices[-1])] = 1

    prediction = rescale(prediction.astype(float), 1 / rescale_coeff)

    # ------ Saving results ------- #
    results = {}

    results['prediction'] = prediction

    with open(path_my_data + '/results.pkl', 'wb') as handle:
        pickle.dump(results, handle)

    imsave(path_my_data + '/AxonDeepSeg.png', prediction, 'png')


# ---------------------------------------------------------------------------------------------------------

def myelin(path_my_data):
    """
    :param path_my_data: folder of the data, must include the segmentation results results.pkl
    :return: no return
    Myelin is Segmented by the AxonSegmentation Toolbox (NeuroPoly)
    The segmentation mask of the myelin is saved in the folder of the data

    TODO : PLEASE DELETE ME ALDO OR ANYBODY ELSE !!!!

    
    """

    file = open(path_my_data + '/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())

    file = open(path_my_data + "/results.pkl", 'r')
    results = pickle.load(file)

    print '\n\n ---START MYELIN DETECTION---'

    io.savemat(path_my_data + '/AxonMask.mat', mdict={'prediction': results["prediction_mrf"]})
    current_path = os.path.dirname(os.path.abspath(__file__))
    print current_path

    command = path_matlab + "/bin/matlab -nodisplay -nosplash -r \"clear all;addpath(\'" + current_path + "\');" \
                                                                                                          "addpath(genpath(\'" + path_axonseg + "/code\')); myelin(\'%s\',%s);exit()\"" % (
    path_my_data, pixel_size)
    os.system(command)

    os.remove(path_my_data + '/AxonMask.mat')


def pipeline(path_my_data, path_model, config, visualize=False):
    """
    :param path_my_data: : folder of the data, must include image.jpg
    :param path_model :  folder of the model of segmentation. Must contain model.ckpt
    :param path_mrf: folder of the mrf parameters.  Must contain mrf_parameter.pkl
    :param visualize: if True, visualization of the results is runned. (and If a groundtruth is in image_path, scores are calculated)
    :return:
    """

    print '\n\n ---START AXON-MYELIN SEGMENTATION---'
    axon_segmentation(path_my_data, path_model, config)
    myelin(path_my_data)
    print '\n End of the process : see results in : ', path_my_data

    if visualize:
        from visualization.visualize import visualize_segmentation
        visualize_segmentation(path_my_data)

# To Call the training in the terminal

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-p", "--path_data", required=True, help="")
    ap.add_argument("-c", "--config_file", required=False, help="", default="~/.axondeepseg.json")

    args = vars(ap.parse_args())
    path_model = args["path_model"]
    path_data = args["path_data"]
    config_file = args["config_file"]

    config = generate_config(config_file)

    axon_segmentation(path_data, path_model, config)


if __name__ == '__main__':
    main()
