
# -*- coding: utf-8 -*-

# Basic integrity test to check is AxonDeepSeg is correctly installed.
# Launches a segmentation in the data_test folder.

import json
import os
from AxonDeepSeg.testing.segmentation_scoring import *
from time import time
from AxonDeepSeg.apply_model import axon_segmentation
from scipy.misc import imread, imsave
import AxonDeepSeg.ads_utils

def integrity_test():

    try:

        # get path of directory where AxonDeepSeg was installed
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # input parameters

        path = os.path.join('folder_name', 'file_name')
        path_testing = os.path.join(dir_path, 'data_test')
        model_name = 'default_SEM_model_v1'
        path_model = os.path.join(dir_path, 'models',model_name)
        path_configfile = os.path.join(path_model, 'config_network.json')

        # Read the configuration file 
        print('Reading test configuration file.')
        if not os.path.exists(path_model):
            os.makedirs(path_model)

        with open(path_configfile, 'r') as fd:
            config_network = json.loads(fd.read())

        # Launch the axon and myelin segmentation on test image sample provided in the installation
        print('Computing the segmentation of axon and myelin on test image.')
        prediction = axon_segmentation([path_testing], ["image.png"], path_model, config_network, prediction_proba_activate=True, verbosity_level=4)

        # Read the ground truth mask and the obtained segmentation mask
        mask = imread(path_testing + '/mask.png', flatten=True)
        pred = imread(path_testing + '/AxonDeepSeg.png', flatten=True)

        # Generate separate axon and myelin masks of the segmentation output
        print('Generating axon and myelin segmentation masks and saving.')
        gt_axon = mask > 200 # Generate binary image with the axons for the ground truth (myelin=127, axon=255)
        gt_myelin = np.logical_and(mask >= 50, mask <= 200) # Generate binary image with the myelin for the ground truth (myelin=127, axon=255)

        pred_axon = pred > 200 # Generate binary image with the axons for the segmentation (myelin=127, axon=255)
        pred_myelin = np.logical_and(pred >= 50, pred <= 200) # Generate binary image with the myelin for the segmentation (myelin=127, axon=255)

        # Compute Dice between segmentation and ground truth, for both axon and myelin
        dice_axon = pw_dice(pred_axon, gt_axon)
        dice_myelin = pw_dice(pred_myelin, gt_myelin)

        # If all the commands above are executed without bugs, the installation is done correctly
        print("* * * Integrity test passed. AxonDeepSeg is correctly installed. * * * ")
        return 0

    except IOError:

        # Else, there is a problem in the installation
        print("Integrity test failed... ")
        return -1
