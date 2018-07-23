from skimage import measure
from skimage.measure import regionprops
import numpy as np
import pandas as pd
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio
import json
from skimage.morphology import binary_erosion, disk, label

from AxonDeepSeg.testing.segmentation_scoring import *
import AxonDeepSeg.ads_utils

def launch_performance_metrics(path_prediction, path_groundtruth):
    """
    :param path_prediction : path of the prediction, output segmentation of AxonDeepSeg
    :param path_groundtruth : path of the ground truth labelling (gold standard)
    :return: axon_metrics, myelin_metrics  
    """

    # Read segmentation image and get axon/myelin masks
    pred = imread(path_prediction, flatten=True)
    pred_axon = pred > 200
    pred_myelin = np.logical_and(pred >= 50, pred <= 200)
    
    # Read groundtruth mask and get axon/myelin masks
    gt = imread(path_groundtruth, flatten=True)
    gt_axon = gt > 200
    gt_myelin = np.logical_and(gt >= 50, gt <= 200)

    # Get folder path
    path_folder, file_name = os.path.split(path_prediction)

    # Compute pixelwise metrics for axon segmentation
    axon_metrics = Metrics_calculator(pred_axon, gt_axon)
    axon_metrics_array = np.array([axon_metrics.pw_sensitivity(),axon_metrics.pw_specificity(),axon_metrics.pw_precision(),
    axon_metrics.pw_accuracy(), axon_metrics.pw_F1_score(), axon_metrics.pw_dice(), axon_metrics.pw_jaccard()])

    # Compute element-wise metrics for axon segmentation
    dice_output = axon_metrics.ew_dice('short')

    # Compute pixelwise metrics for myelin segmentation
    myelin_metrics = Metrics_calculator(pred_myelin, gt_myelin)
    myelin_metrics_array = np.array([myelin_metrics.pw_sensitivity(),myelin_metrics.pw_specificity(),myelin_metrics.pw_precision(),
    myelin_metrics.pw_accuracy(), myelin_metrics.pw_F1_score(), myelin_metrics.pw_dice(),myelin_metrics.pw_jaccard()])

    return axon_metrics, myelin_metrics  
