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
from AxonDeepSeg.train_network_tools import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from config_tools import update_config, default_configuration


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




## New version of im 2 patches

def im2patches(img, crop_value=25, scw=256):
    # First we crop the image to get the context
    cropped = img[crop_value:-crop_value, crop_value:-crop_value]
    
    # Then we create patches using the prediction window
    spw = scw-2*crop_value #size prediction windows
    
    qh, rh = divmod(cropped.shape[0], spw)
    qw, rw = divmod(cropped.shape[1], spw)
    
    # Creating positions of prediction windows
    L_h = [spw*e for e in range(qh)]
    L_w = [spw*e for e in range(qw)]
    
    # Then if there is a remainder we take the last positions (overlap on the last predictions)
    if rh != 0:
        L_h.append(cropped.shape[0] - spw)
    if rw != 0:
        L_w.append(cropped.shape[1] - spw)
    
    xx, yy = np.meshgrid(L_h, L_w)
    P = [np.ravel(xx), np.ravel(yy)]
    L_pos = [[P[0][i], P[1][i]] for i in range(len(P[0]))]
    
    # These positions are also the positions of the context windows in the base image coordinates !
    L_patches = []
    for e in L_pos:
        patch = img[e[0]:e[0]+scw,e[1]:e[1]+scw]
        patch = exposure.equalize_hist(patch)
        patch = (patch - np.mean(patch)) / np.std(patch)
        L_patches.append(patch)
                         
    return [img, L_patches, L_pos]


def patches2im(L_patches, L_pos, cropped_value = 25, scw=256):
    spw = scw-2*cropped_value
    #L_pred = [e[cropped_value:-cropped_value,cropped_value:-cropped_value] for e in L_patches]
    # First : extraction of the predictions
    h_l, w_l = np.max(np.stack(L_pos), axis=0)
    L_pred = []
    new_img = np.zeros((h_l+scw,w_l+scw))
    for i,e in enumerate(L_patches):
        if L_pos[i][0] == 0:
            if L_pos[i][1] == 0:
                new_img[0:cropped_value,0:cropped_value] = e[0:cropped_value,0:cropped_value]
                new_img[cropped_value:scw-cropped_value, 0:cropped_value] = e[cropped_value:-cropped_value,0:cropped_value]
                new_img[0:cropped_value, cropped_value:scw-cropped_value] = e[0:cropped_value,cropped_value:-cropped_value]
            else:
                if L_pos[i][1] == w_l:
                    new_img[0:cropped_value,-cropped_value:] = e[0:cropped_value,-cropped_value:]
                new_img[0:cropped_value,L_pos[i][1]+cropped_value:L_pos[i][1]+scw-cropped_value] = e[0:cropped_value,cropped_value:-cropped_value]
                
        if L_pos[i][1] == 0:
            if L_pos[i][0] != 0:
                new_img[L_pos[i][0]+cropped_value:L_pos[i][0]+scw-cropped_value, 0:cropped_value] = e[cropped_value:-cropped_value, 0:cropped_value]
        
        if L_pos[i][0] == h_l:
            if L_pos[i][1] == w_l:
                new_img[-cropped_value:,-cropped_value:] = e[-cropped_value:,-cropped_value:]
                new_img[h_l+cropped_value:-cropped_value, -cropped_value:] = e[cropped_value:-cropped_value,-cropped_value:]
                new_img[-cropped_value:, w_l+cropped_value:-cropped_value] = e[-cropped_value:,cropped_value:-cropped_value]
            else:
                if L_pos[i][1] == 0:
                    new_img[-cropped_value:,0:cropped_value] = e[-cropped_value:,0:cropped_value]

                    
                new_img[-cropped_value:,L_pos[i][1]+cropped_value:L_pos[i][1]+scw-cropped_value] = e[-cropped_value:,cropped_value:-cropped_value]
        if L_pos[i][1] == w_l:
            if L_pos[i][1] != h_l:
                new_img[L_pos[i][0]+cropped_value:L_pos[i][0]+scw-cropped_value, -cropped_value:] = e[cropped_value:-cropped_value, -cropped_value:]

                
    L_pred = [e[cropped_value:-cropped_value,cropped_value:-cropped_value] for e in L_patches]
    L_pos_corr = [[e[0]+cropped_value, e[1]+cropped_value] for e in L_pos]
    for i,e in enumerate(L_pos_corr):
        new_img[e[0]:e[0]+spw, e[1]:e[1]+spw] = L_pred[i]
        
    return new_img


def apply_convnet(path_my_data, path_model, config, batch_size=1, crop_value=25, general_pixel_size=0.2):
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
    img = rescale(img_org, rescale_coeff, preserve_range=True).astype(int)

    ###############
    # Network Parameters
    patch_size = config["network_trainingset_patchsize"]
    thresh_indices = config["network_thresholds"]
    ##############

    folder_model = path_model
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    # --------------------SAME ALGORITHM IN TRAIN_model---------------------------

    x = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size))

    ####################################################

    # Call the model
    pred = uconv_net(x, config, phase=False, verbose=False)
    saver = tf.train.Saver(tf.global_variables())

    # Image to batch
    image_init, data, positions = im2patches(img, crop_value, patch_size)
    predictions = []
    
    # Limit the size
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    

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
        Mask = np.argmax(p,axis=2)    
        Mask = Mask.reshape(batch_size, patch_size,patch_size) # Now Mask is a 256*256 mask with Mask[i,j] = pixel_class
        predictions.extend([np.squeeze(e) for e in np.split(Mask, batch_size, axis=0)])
        
    # Last batch
    if rem != 0:
        #print 'processing last patch'
        batch_x = np.asarray(data[it*batch_size:])
        p = sess.run(pred, feed_dict={x: batch_x}) # here we get the predictions under prob format (float, between 0 and 1, shape = (bs, size_image*size_image, n_classes).          
        Mask = np.argmax(p,axis=2)    
        Mask = Mask.reshape(rem, patch_size,patch_size) # Now Mask is a 256*256 mask with Mask[i,j] = pixel_class
        predictions.extend([np.squeeze(e) for e in np.split(Mask, rem, axis=0)])
        
    sess.close()
    tf.reset_default_graph()

    # -----------------------Merge each segmented patch to reconstruct the total segmentation

    h_size, w_size = image_init.shape
    prediction_rescaled = patches2im(predictions, positions, crop_value, patch_size)
    #labellize_mask_2d()
    prediction = resize(prediction_rescaled, img_org.shape)
    prediction = prediction.astype(np.uint8) # Rescaling operation can change the vlue of the pixels to float.

    # Image returned is of same shape as total image and with each pixel being the class it's been attributed to   
    return prediction

    #######################################################################################################################

def axon_segmentation(path_my_data, path_model, config, imagename = 'AxonDeepSeg.png', crop_value = 25, batch_size=1, general_pixel_size=0.2):
    """
    :param path_my_data: folder of the image to segment. Must contain image.jpg
    :param path_model: folder of the model of segmentation. Must contain model.ckpt
    :return: no return
    Results including the prediction and the prediction associated with the mrf are saved in the image_path
    AxonMask.mat is saved in the image_path to feed the detection of Myelin
    /AxonSeg.jpeg is saved in the image_path
    """
    
    # Ensuring that the config file is alright
    config = update_config(default_configuration(), config)

    file = open(path_my_data + '/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())
    rescale_coeff = pixel_size / general_pixel_size
    
    path_img = path_my_data + '/image.png' 
    img = imread(path_img, flatten=False, mode='L')

    # ------ Apply ConvNets ------- #
    prediction = apply_convnet(path_my_data, path_model, config, batch_size, crop_value = crop_value, general_pixel_size = general_pixel_size) # Predictions are shape of image, value = class of pixel
    
    # We now transform the prediction to an image
    n_classes = config['network_n_classes']
    paint_vals = [int(255*float(i)/(n_classes - 1)) for i in range(n_classes)]
    
    
    # Now we create the mask with values in range 0-255
    mask = np.zeros_like(prediction)
    for i in range(n_classes):
        mask[prediction == i] = paint_vals[i]
            
    # ------ Saving results ------- #
    results = {}

    results['prediction'] = prediction

    #with open(path_my_data + '/results.pkl', 'wb') as handle:
    #    pickle.dump(results, handle)

    imsave(path_my_data + '/'+imagename, mask, 'png')


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
