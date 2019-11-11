# coding: utf-8


import numpy as np
from skimage import io
import os
import imageio
import json
from skimage import transform
from skimage.filters import gaussian

from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.testing.segmentation_scoring import *
from imageio import imread, imsave

from AxonDeepSeg.testing.segmentation_scoring import *

import AxonDeepSeg.ads_utils

def add_additive_gaussian_noise(img,mu=0,sigma=10):
	'''
	:param img: input image to add noise on
	:param mu: mean of the gaussian noise model
	:param sigma: sigma of the gaussian noise model
	:return: img_noise, the noisy image
	'''
	
	# Generate random gaussian noise with specified mean and sigma values
	noise = np.random.normal(mu, sigma, img.shape)
	
	# Add generated noise to input image
	img_noise = np.add(img,noise)

	# Clip noisy image between 0-255
	img_noise[img_noise < 0] = 0
	img_noise[img_noise > 255] = 255

	return img_noise.astype(np.uint8)


def add_multiplicative_gaussian_noise(img,mu=1,sigma=0.05):
	'''
	:param img: input image to add noise on
	:param mu: mean of the gaussian noise model
	:param sigma: sigma of the gaussian noise model
	:return: img_noise, the noisy image
	'''

	# Generate random gaussian noise with specified mean and sigma values
	noise = np.random.normal(mu, sigma, img.shape)
	
	# Add generated noise to input image
	img_noise = np.multiply(img,noise)

	# Clip noisy image between 0-255
	img_noise[img_noise < 0] = 0
	img_noise[img_noise > 255] = 255

	return img_noise.astype(np.uint8)


def change_brightness(img,value_percentage=0.2):
	'''
	:param img: input image to add noise on
	:param value_percentage: % of change in brightness (positive=increase brightness, negative=decrease)
	:param sigma: sigma of the gaussian noise model
	:return: img_noise, the noisy image
	'''

	# Add generated noise to input image
	img_noise = img+(value_percentage*255)

	# Clip noisy image between 0-255
	img_noise[img_noise < 0] = 0
	img_noise[img_noise > 255] = 255

	return img_noise.astype(np.uint8)
