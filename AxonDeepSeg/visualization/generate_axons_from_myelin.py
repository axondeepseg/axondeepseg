
# -*- coding: utf-8 -*-

# Function to generate corrected axon+myelin image after correcting the myelin mask.

import os
import numpy as np

import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.ads_utils import convert_path

def generate_axons_from_myelin(path_prediction,path_myelin_corrected):
    """
    :param path_prediction: path of the prediction i.e. image of axon+myelin segmentation (output of AxonDeepSeg)
    :param path_myelin_corrected: path of corrected myelin by the user i.e. myelin mask (uint8 type with myelin=255, background=0)
    :return: merged and corrected axon+myelin image
    """

    # If string, convert to Path objects
    path_prediction = convert_path(path_prediction)
    path_myelin_corrected = convert_path(path_myelin_corrected)

    # read output from axondeepseg and myelin mask corrected by user
    prediction = ads.imread(path_prediction)
    myelin_corrected = ads.imread(path_myelin_corrected)

    # compute the axon mask from axondeepseg (axon=255, myelin=127, background=0)
    axon_ads = prediction > 200

    # get the myelin mask corrected by user (myelin=255, background=0)
    myelin_corrected = myelin_corrected > 200

    # compute logical OR between axondeepseg axon mask and myelin corrected mask
    fused = np.logical_or(axon_ads, myelin_corrected)

    # compute new axon mask by logical XOR between corrected myelin mask and fused
    new_axon_mask = np.logical_xor(myelin_corrected, fused)

    # merge corrected myelin mask and generated axon mask
    both = new_axon_mask*255 + myelin_corrected*127

    # save the corrected axon+myelin image
    path = path_prediction.parent / 'axon_myelin_mask_corrected.png'
    ads.imwrite(path, both)

    return both
