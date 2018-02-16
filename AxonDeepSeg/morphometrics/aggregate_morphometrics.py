# coding: utf-8

import numpy as np
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio
import json
from skimage import transform
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from shutil import copy
import math
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.testing.segmentation_scoring import *


from scipy.misc import imread, imsave
from skimage import measure
from skimage.measure import regionprops



def get_aggregate_morphometrics(pred_axon, pred_myelin):
	'''
    :param pred_axon: axon mask from axondeepseg segmentation output
    :param pred_myelin: myelin mask from axondeepseg segmentation output
    :return: aggregate_metrics: dictionary containing values of aggregate metrics
    '''
    
    # Compute AVF (axon volume fraction) = area occupied by axons in sample
    avf = np.count_nonzero(pred_axon)/float((pred_axon.size))
    # Compute MVF (myelin volume fraction) = area occupied by myelin sheaths in sample
    mvf = np.count_nonzero(pred_myelin)/float((pred_myelin.size))
    
    # Estimate aggregate g-ratio = sqrt(1/(1+MVF/AVF))
    gratio = math.sqrt(1/(1+(float(mvf)/float(avf))))
    
    # Get individual axons metrics and compute mean axon diameter
    stats_array = get_axon_morphometrics(pred_axon)
    axon_diam_list = [d['axon_diam'] for d in stats_array]
    mean_axon_diam = np.mean(axon_diam_list)
    
    # Estimate mean myelin diameter (axon+myelin diameter) by using aggregate g-ratio = mean_axon_diam/mean_myelin_diam
    mean_myelin_diam = mean_axon_diam/gratio
    
    # Estimate mean myelin thickness = mean_myelin_radius - mean_axon_radius
    mean_myelin_thickness = (float(mean_myelin_diam)/2) - (float(mean_axon_diam)/2)
    
    # Compute axon density (number of axons per mm2)
    y, x = pred_axon.shape
    img_area_mm2 = float(pred_axon.size)*get_pixelsize(os.path.join(path_folder,'pixel_size_in_micrometer.txt'))*get_pixelsize(os.path.join(path_folder,'pixel_size_in_micrometer.txt'))/(float(1000000))
    axon_density_mm2 = float(len(axon_diam_list))/float(img_area_mm2)
    
    # Create disctionary to store aggregate metrics
    aggregate_metrics = {'avf': avf, 'mvf': mvf, 'gratio': gratio, 'mean_axon_diam': mean_axon_diam,
                         'mean_myelin_diam': mean_myelin_diam,'mean_myelin_thickness': mean_myelin_thickness,
                         'axon_density_mm2': axon_density_mm2}

    return aggregate_metrics


def write_aggregate_morphometrics(path_folder,aggregate_metrics):
	'''
    :param path_folder: absolute path of folder containing sample + segmentation
    :param aggregate_metrics: dictionary containing values of aggregate metrics
    :return: nothing
    '''   
    f = open(os.path.join(path_folder,'aggregate_morphometrics.txt'), 'w')
    f.write('aggregate_metrics: ' + repr(aggregate_metrics) + '\n')
    f.close()

