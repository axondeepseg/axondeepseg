# -*- coding: utf-8 -*-

import math
import os
import tensorflow as tf
import pickle
import numpy as np
from scipy import io
from scipy.misc import imread, imsave
from skimage.transform import rescale, resize
from skimage import exposure
from config_tools import general_pixel_size, path_matlab, path_axonseg, generate_config
from AxonDeepSeg.train_network_tools import *
from tqdm import tqdm


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



def apply_convnet(path_my_data, path_model, config, batch_size=1, pred_proba=False):
    """
    :param path_my_data: folder of the image to segment. Must contain image.png
    :param path_model: folder of the model of segmentation. Must contain model.ckpt
    :param config: dict: network's parameters described in the header.
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: prediction, the mask of the segmentation
    """
    from logging import WARN
    tf.logging.set_verbosity(WARN)

    #print '\n\n ---Start axon segmentation on %s---' % path_my_data

    path_img = path_my_data + '/image.png'
    img_org = imread(path_img, flatten=False, mode='L')

    file = open(path_my_data + '/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())

    # set the resolution to the general_pixel_size
    rescale_coeff = pixel_size / general_pixel_size
    img = (rescale(img_org, rescale_coeff) * 256).astype(int)

    ###############
    # Network Parameters
    image_size = 256
    thresh_indices = config["network_thresholds"]
    n_classes = config["network_n_classes"]
    ##############

    folder_model = path_model
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    if os.path.exists(folder_model + '/hyperparameters.pkl'):
        #print 'hyperparameters detected in the model'
        hyperparameters = pickle.load(open(folder_model + '/hyperparameters.pkl', "rb"))
        image_size = hyperparameters['image_size']

    # --------------------SAME ALGORITHM IN TRAIN_model---------------------------

    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size))

    ####################################################

    # Call the model
    pred = uconv_net(x, config, phase=False, verbose=False)

    saver = tf.train.Saver(tf.global_variables())

    # Image to batch
    image_init, data, positions = im2patches(img, 256)
    predictions_list = []
    predictions_proba_list = []
    
    # Limit the size
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    

    # Launch the graph
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.import_meta_graph(path_model + '/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(path_model))

    # --------------------- Apply the segmentation to each patch of the images--------------------------------
    n_patches = len(data)
    it, rem = divmod(n_patches, batch_size)

    for i in tqdm(range(it)):
        #print 'processing patch %s on %s' % (i+1, it)
        batch_x = np.asarray(data[i*batch_size:(i+1)*batch_size])
        p = sess.run(pred, feed_dict={x: batch_x}) # here we get the predictions under prob format (float, between 0 and 1, shape = (bs, size_image*size_image, n_classes).          
        # Generating the mask for each element of the batch
        Mask = np.argmax(p,axis=2)
        Mask = Mask.reshape(batch_size, 256,256) # Now Mask is a 256*256 mask with Mask[i,j] = pixel_class
        predictions_list.extend(np.split(Mask, batch_size, axis=0))

        if pred_proba:
            # Generating the probas for each element of the batch (basically changing the shape of the prediction)
            p = p.reshape(batch_size, 256, 256, n_classes)
            predictions_proba_list.extend(np.split(p, batch_size, axis=0))

    # Last batch
    if rem != 0:
        #print 'processing last patch'
        batch_x = np.asarray(data[it*batch_size:])
        p = sess.run(pred, feed_dict={x: batch_x}) # here we get the predictions under prob format (float, between 0 and 1, shape = (bs, size_image*size_image, n_classes).          
        Mask = np.argmax(p,axis=2)    
        Mask = Mask.reshape(rem, 256,256) # Now Mask is a 256*256 mask with Mask[i,j] = pixel_class
        predictions_list.extend(np.split(Mask, rem, axis=0))
        if pred_proba:
            # Generating the probas for each element of the batch (basically changing the shape of the prediction)
            p = p.reshape(rem, 256, 256, n_classes)
            predictions_proba_list.extend(np.split(p, rem, axis=0))

    sess.close()
    tf.reset_default_graph()

    # -----------------------Merge each segmented patch to reconstruct the total segmentation

    h_size, w_size = image_init.shape

    # Then we stitch the different images
    prediction_stitched = patches2im(predictions_list, positions, h_size, w_size)
    prediction = resize(prediction_stitched, img_org.shape)
    prediction = prediction.astype(np.uint8) # Rescaling operation can change the value of the pixels to float.
    if pred_proba:
        # We generate the predict proba matrix

        tmp = np.split(np.stack(predictions_proba_list, axis=0), n_classes, axis=-1)
        predictions_proba_list = [np.split(e, n_patches, axis=0) for e in
                             tmp]  # We now have a list (n_classes elements) of list (n_patches elements)
        prediction_proba_stitched = [patches2im(e, positions, h_size, w_size) for e in predictions_proba_list]
        prediction_proba = np.stack([resize(e, img_org.shape) for e in prediction_proba_stitched], axis=-1)

    # Image returned is of same shape as total image and with each pixel being the class it's been attributed to
        return prediction, prediction_proba
    else:

        print prediction.shape
        return prediction

    #######################################################################################################################

def axon_segmentation(path_my_data, path_model, config, imagename = 'AxonDeepSeg.png', batch_size=1, pred_proba=False):
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
    
    path_img = path_my_data + '/image.png' 
    img = imread(path_img, flatten=False, mode='L')

    # ------ Apply ConvNets ------- #
    if pred_proba:
        prediction, prediction_proba = apply_convnet(path_my_data, path_model, config, batch_size, pred_proba=pred_proba) # Predictions are shape of image, value = class of pixel
    else:
        prediction = apply_convnet(path_my_data, path_model, config,
                                                     batch_size)  # Predictions are shape of image, value = class of pixel
    
    # We now transform the prediction to an image
    n_classes = config['network_n_classes']
    paint_vals = [int(255*float(i)/(n_classes - 1)) for i in range(n_classes)]
    
    
    # Now we create the mask with values in range 0-255
    mask = np.zeros_like(prediction)
    for i in range(n_classes):
        mask[prediction == i] = paint_vals[i]
            
    # ------ Saving results ------- #
    #results = {}

    #results['prediction'] = prediction

    #with open(path_my_data + '/results.pkl', 'wb') as handle:
    #    pickle.dump(results, handle)

    imsave(path_my_data + '/'+imagename, mask, 'png')
    if pred_proba:
        return prediction, prediction_proba
    else:
        return prediction

# ---------------------------------------------------------------------------------------------------------


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
    ap.add_argument("-n", "--imagename", required=False, help="", default="AxonDeepSeg.png")
    ap.add_argument("-c", "--config_file", required=False, help="", default=None)

    args = vars(ap.parse_args())

    path_model = args["path_model"]
    path_data = args["path_data"]
    config_file = args["config_file"]

    # We can't use the default argument from argparse.add_argument as it relies on another argument being entered (path_model).
    # Instead, we set it ourselves below if path_config has not been given.
    if config_file == None:
        config_file = path_model + '/config_network.json'
    imagename = args["imagename"]

    config = generate_config(config_file)

    axon_segmentation(path_data, path_model, config, imagename = imagename)



if __name__ == '__main__':
    main()
