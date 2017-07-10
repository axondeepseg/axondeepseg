#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from AxonDeepSeg.apply_model import axon_segmentation

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

			axon_segmentation(path_datatest, subpath_model, config_network, imagename='segmentation_' + root + '.png')

    return 'segmented'

def map_model_to_images(folder_model, path_datatests):
    """
    Apply one trained model to all the specified images
    """

    for root in os.listdir(path_datatests):
        if 'DS_Store' not in root:
            # Subpath image to apply
            subpath_image = os.path.join(path_datatests, root)

            # Load config
            with open(folder_model + 'config_network.json', 'r') as fd:
                config_network = json.loads(fd.read())

            axon_segmentation(subpath_image, folder_model, config_network, imagename='segmentation_' + root + '.png')
            print 'Segmentation ' + str(root) + ' done.'






