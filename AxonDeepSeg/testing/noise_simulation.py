# coding: utf-8

import numpy as np
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio
import json
from skimage import transform
from skimage.filters import gaussian

from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.testing.segmentation_scoring import *
from scipy.misc import imread, imsave


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



def add_gaussian_blurring(img,sigma=10):
	'''
	:param img: input image to add noise on
	:param mu: mean of the gaussian noise model
	:param sigma: sigma of the gaussian noise model
	:return: img_noise, the noisy image
	'''
	
	# Generate random gaussian noise with specified mean and sigma values
	img_noise = gaussian(img, sigma)
	img_noise = img_noise*255
	
	return img_noise.astype(np.uint8)


def reduce_contrast(img,blending_factor=0.1):
	'''
	:param img: input image to add noise on
	:param mu: mean of the gaussian noise model
	:param sigma: sigma of the gaussian noise model
	:return: img_noise, the noisy image
	'''
	y, x = img.shape
	gray_img = np.mean(img)*np.ones((y, x))
	img_noise = img*(1-blending_factor)+gray_img*blending_factor

	return img_noise.astype(np.uint8)

def increase_brightness(img,blending_factor=0.2):
	'''
	:param img: input image to add noise on
	:param value_percentage: % of change in brightness (positive=increase brightness, negative=decrease)
	:param sigma: sigma of the gaussian noise model
	:return: img_noise, the noisy image
	'''

	y, x = img.shape
	gray_img = 255*np.ones((y, x))
	img_noise = img*(1-blending_factor)+gray_img*blending_factor

	return img_noise.astype(np.uint8)


def decrease_brightness(img,blending_factor=0.2):
	'''
	:param img: input image to add noise on
	:param value_percentage: % of change in brightness (positive=increase brightness, negative=decrease)
	:param sigma: sigma of the gaussian noise model
	:return: img_noise, the noisy image
	'''

	y, x = img.shape
	gray_img = np.zeros((y, x))
	img_noise = img*(1-blending_factor)+gray_img*blending_factor

	return img_noise.astype(np.uint8)







