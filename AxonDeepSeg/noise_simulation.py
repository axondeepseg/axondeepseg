
# -*- coding: utf-8 -*-

# Utility functions for noise simulation

import numpy as np
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio
import json
import matplotlib.pyplot as plt

def add_additive_gaussian_noise(img,mu=0,sigma=10):

    # Generate random gaussian noise with specified mean and sigma values
    noise = np.random.normal(mu, sigma, img.shape)
    
    # Add generated noise to input image
    img_noise = np.add(img,noise)

    # Clip noisy image between 0-255
    img_noise[img_noise < 0] = 0
    img_noise[img_noise > 255] = 255

    return img_noise.astype(np.uint8)



def add_multiplicative_gaussian_noise(img,mu=1,sigma=0.05):

    # Generate random gaussian noise with specified mean and sigma values
    noise = np.random.normal(mu, sigma, img.shape)
    
    # Add generated noise to input image
    img_noise = np.multiply(img,noise)

    # Clip noisy image between 0-255
    img_noise[img_noise < 0] = 0
    img_noise[img_noise > 255] = 255

    return img_noise.astype(np.uint8)



def change_illumination(img,value_percentage=0.2):

    # Add generated noise to input image
    img_noise = img+(value_percentage*255)

    # Clip noisy image between 0-255
    img_noise[img_noise < 0] = 0
    img_noise[img_noise > 255] = 255

    return img_noise.astype(np.uint8)








