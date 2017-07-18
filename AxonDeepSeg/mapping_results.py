#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from AxonDeepSeg.apply_model import axon_segmentation
from tqdm import tqdm

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

def map_model_to_images(folder_model, path_datatests, batch_size=1):
    """
    Apply one trained model to all the specified images
    """

    for root in tqdm(os.listdir(path_datatests), desc='images'):
        if 'DS_Store' not in root and 'txt' not in root:
            # Subpath image to apply
            subpath_image = os.path.join(path_datatests, root)

            # Load config
            with open(os.path.join(folder_model, 'config_network.json'), 'r') as fd:
                config_network = json.loads(fd.read())

            axon_segmentation(subpath_image, folder_model, config_network, imagename='segmentation_' + root + '.png', batch_size=batch_size)
            
            file = open(path_datatests + "/report.txt", 'a')
            output_text = str(root) + ' done ..\n'
            file.write(output_text)
            file.close()
            #print 'Segmentation ' + str(root) + ' done.'






