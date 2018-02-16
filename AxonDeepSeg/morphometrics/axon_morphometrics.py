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


def get_pixelsize(path_pixelsize_file):
	'''
    :param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
    :return: the pixel size value.
    '''

    text_file = open(path_pixelsize_file, "r")
    pixelsize = float(text_file.read())
    text_file.close()
    return pixelsize



def get_axon_morphometrics(pred_axon):
	'''
    :param pred_axon: axon binary mask, output of axondeepseg
    :return: list of dictionaries containing for each axon, various morphometrics
    '''

    # Array for keeping axon-wise metrics
    stats_array = np.empty(0)

    # Label each axon object
    labels = measure.label(pred_axon)
    axon_objects = regionprops(labels)

    # Get axon morphometrics of interest
    for props in axon_objects:
    
        # Centroid
        y0, x0 = props.centroid
        # Solidity
        solidity = props.solidity
        # Eccentricity
        eccentricity = props.eccentricity
        # Axon equivalent diameter in micrometers
        axon_diam = (props.equivalent_diameter)*get_pixelsize(os.path.join(path_folder,'pixel_size_in_micrometer.txt'))
        # Axon orientation angle
        orientation = props.orientation
    
        # Add metrics to list of dictionaries
        stats = {'y0': y0, 'x0': x0, 'axon_diam': axon_diam, 'solidity':solidity,'eccentricity': eccentricity,
                 'orientation':orientation}
        stats_array = np.append(stats_array, [stats], axis=0)
        
    return stats_array


def save_axon_morphometrics(path_folder,stats_array):
	'''
    :param path_folder: absolute path of folder containing the sample + the segmentation output
    :param stats_array: list of dictionaries containing axon morphometrics
    :return: nothing
    '''
    np.save(os.path.join(path_folder,'axonlist.npy'),stats_array)


def load_axon_morphometrics(path_folder):
	'''
    :param path_folder: absolute path of folder containing the sample + the segmentation output
    :return: stats_array: list of dictionaries containing axon morphometrics
    '''
    stats_array = np.load(os.path.join(path_folder,'axonlist.npy'))
    return stats_array


def display_axon_diameter(img,path_folder,pred_axon,pred_myelin):
	'''
	:param img: sample grayscale image (png)
    :param path_folder: absolute path of folder containing sample + segmentation
    :param aggregate_metrics: dictionary containing values of aggregate metrics
    :param pred_axon: axon mask from axondeepseg segmentation output
    :param pred_myelin: myelin mask from axondeepseg segmentation output
    :return: nothing
    '''   
    stats_array = get_axon_morphometrics(pred_axon)
    axon_diam_list = [d['axon_diam'] for d in stats_array]
    axon_diam_array = np.asarray(axon_diam_list)
    axon_iter = np.arange(np.size(axon_diam_array))

    labels = measure.label(pred_axon)
    axon_diam_display = a = np.zeros((np.shape(labels)[0], np.shape(labels)[1]))

    for pix_x in np.arange(np.shape(labels)[0]):
           for pix_y in np.arange(np.shape(labels)[1]):
                if labels[pix_x,pix_y] != 0:
                    axon_diam_display[pix_x,pix_y] = axon_diam_array[labels[pix_x,pix_y]-1]
        
    # Axon display
    plt.figure(figsize=(12,9))
    im = plt.imshow(axon_diam_display,cmap='hot')
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.title('Axon display colorcoded with axon diameter in um',fontsize=12)           
    plt.savefig(os.path.join(path_folder,'display_axon_diameter.png'))   
    
    # Axon overlay on original image
    plt.figure(figsize=(12,9))
    plt.imshow(img, cmap='gray', alpha=0.8)
    im = plt.imshow(axon_diam_display, cmap='hot', alpha=0.5)
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.title('Axon overlay colorcoded with axon diameter in um',fontsize=12)
    plt.savefig(os.path.join(path_folder,'overlay_axon_diameter.png')) 
    
    # Axon overlay on original image + myelin display (same color for every myelin sheath)
    plt.figure(figsize=(12,9))
    plt.imshow(img, cmap='gray', alpha=0.8)
    plt.imshow(pred_myelin,cmap='gray', alpha=0.2)
    im = plt.imshow(axon_diam_display, cmap='hot', alpha=0.5)
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.savefig(os.path.join(path_folder,'overlay_axon_diameter_with_myelin.png')) 



