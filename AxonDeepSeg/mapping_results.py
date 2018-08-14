#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from AxonDeepSeg.apply_model import axon_segmentation
from tqdm import tqdm
import AxonDeepSeg.ads_utils

# FILE TO UPDATE

def result_mapping(folder_models, path_datatest):
    """
    Create the U-net.
    Input :
        folder_models : string : path to the folder containing all the models folders to test.
        path_datatest : string : path to the folder containing the image to segment.

    Output :
        None. Save the segmented images in the data_test folder.
    """

    for root in os.listdir(folder_models)[:]:
        if 'DS_Store' not in root:
            subpath_model = os.path.join(folder_models, root)
            filename = '/config_network.json'
            with open(subpath_model + filename, 'r') as fd:
                config_network = json.loads(fd.read())

            axon_segmentation(path_datatest, subpath_model, config_network, segmentations_filenames='segmentation_' + root + '.png')

    return 'segmented'

def map_model_to_images(folder_model, path_datatests, batch_size=1, gps=0.1, crop_value=25, gpu_per=1.0):
    """
    Apply one trained model to all the specified images
    """

    # Load config
    with open(os.path.join(folder_model, 'config_network.json'), 'r') as fd:
        config_network = json.loads(fd.read())
    
    path_images = [os.path.join(path_datatests,e) for e in os.listdir(path_datatests)[:] if os.path.isdir(os.path.join(path_datatests,e))]
    n_images = len(path_images)
    path_images_list = list(segment_list(path_images,20))
    
    if type(gps) != list:
        gps = n_images*[gps]
    gps_list = list(segment_list(gps,20))
    
    for i,path_images_iter in enumerate(path_images_list):
        gps_iter = gps_list[i]
        axon_segmentation(path_images_iter, folder_model, config_network, segmentations_filenames='segmentation.png', inference_batch_size=batch_size, write_mode=True, prediction_proba_activate=False, resampled_resolutions=gps_iter, overlap_value= crop_value, gpu_per=gpu_per)
            
            
def segment_list(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]






