from skimage.transform import rescale
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
from skimage.filters import gaussian
import numpy as np
import random

import AxonDeepSeg.ads_utils
#######################################################################################################################
#                                                Data Augmentation                                                    #
#######################################################################################################################

# We find here the functions that perform transformations to images and mask. 
# All functions take 8-bit images and return 8-bit images


def shifting(patch_size, n_classes, percentage_max = 0.1, verbose=1):
    """
    Shifts the input by a random number of pixels.
    :param patch: List of 2 or 3 ndarrays, [image,mask, (weights)]



    :param percentage_max: Float, maximum value of the shift, in terms of percentage wrt the size of the input.
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List composed of 2 or 3 random shifted the pair [image,mask, (weights)]
    """

    size_shift = int(percentage_max*patch_size) # Maximum shift in pixels.
    
    begin_h = np.random.randint(2*size_shift-1)/ patch_size
    begin_w = np.random.randint(2*size_shift-1)/ patch_size

    
    print(('height shift: ',begin_h, ', width shift: ', begin_w))

    return begin_h, begin_w
    
def rescaling(patch, n_classes, factor_max=1.2, verbose=1):
    """
    Resamples the image by a factor between 1/factor_max and factor_max. Does not resample if the factor is
    too close to 1. Random sampling increases axons size diversity.
    :param patch: List of 2 or 3 ndarrays [image,mask,(weights)]
    :param factor_max: Float, maximum rescaling factor possible. Minimum is obtained by inverting this max_factor.
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List of 2 or 3 randomly rescaled input, [image,mask, (weights)]
    """

    low_bound = 1.0/factor_max
    high_bound = 1.0*factor_max

    # Randomly choosing the resampling factor.
    scale = np.random.uniform(low_bound, high_bound, 1)[0]
    print(('rescaling factor: ', scale))
        
    patch_size = patch[0]
    new_patch_size = int(patch_size*scale)

    # If the resampling factor is too close to 1 we do not resample.
    if (new_patch_size <= patch_size+5) and (new_patch_size >= patch_size-5): # To avoid having q_h = 0
        return None
    else :
       
       return scale
    



def random_rotation(low_bound=5, high_bound=89, verbose=1):
    """
    Rotates randomly the input, angle between low_bound and high_bound.
    :param patch: List of 2 or 3 inputs (ndarrays) [image, mask, (weights)]
    :param low_bound: Int, lower bound of the randomly selected rotation angle.
    :param high_bound: Int, higher bound of the randomly selected rotation angle.
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List of 2 or 3 randomly rotated inputs [image,mask, (weights)]
    """

    # Randomly choosing the angle of rotation.
    angle = np.random.uniform(low_bound, high_bound, 1)
    return angle        




def elastic_transform(patch, alpha, sigma):
    """

    :param patch: List of 2 or 3 ndarrays [image,mask,(weights)]
    :param alpha: deformation coefficient (high alpha -> strong deformation)
    :param sigma: std of the gaussian filter. (high sigma -> smooth deformation)
    :return: List of deformed input [image_deformed, mask_deformed]
    """

    image = patch[0]
    mask = patch[1]
    if len(patch) == 3:
        weights = patch[2]

    random_state = np.random.RandomState(None)
    shape = image.shape

    d = 4
    sub_shape = (shape[0] // d, shape[0] // d)

    deformations_x = random_state.rand(*sub_shape) * 2 - 1
    deformations_y = random_state.rand(*sub_shape) * 2 - 1

    deformations_x = np.repeat(np.repeat(deformations_x, d, axis=1), d, axis = 0)
    deformations_y = np.repeat(np.repeat(deformations_y, d, axis=1), d, axis = 0)

    dx = gaussian_filter(deformations_x, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(deformations_y, sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    elastic_image = map_coordinates(image, indices, order=1).reshape(shape)
    elastic_mask = np.stack([map_coordinates(np.squeeze(e), indices, order=1).reshape(shape)
                           for e in np.split(mask,mask.shape[-1], axis=2)], axis=-1)
    elastic_mask = np.array(elastic_mask)

    if len(patch) == 3:
        elastic_weights = map_coordinates(weights, indices, order=1).reshape(shape)
        elastic_weights = np.array(elastic_weights)

        return [elastic_image.astype(np.uint8), elastic_mask.astype(np.uint8), elastic_weights.astype(np.float32)]
    else:
        return [elastic_image.astype(np.uint8), elastic_mask.astype(np.uint8)]

    
    
    
def elastic(patch, alpha_max=9, verbose=0):
    """
    Elastic transform wrapper for a list of [image, mask]
    :param patch: List of 2 or 3 ndarrays [image,mask,(weights)]
    :param alpha_max: Alpha_max is the maximum value the coefficient of elastic transformation can take. It is randomly
    chosen.
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List of 2 or 3 ndarrays [deformed_image, deformed_mask, (deformed_weights)]
    """

    alpha = random.choice(list(range(1, alpha_max)))
    if verbose>=1:
        print(('elastic transform alpha coeff: ', alpha))
    
    patch_deformed = elastic_transform(patch, alpha = alpha, sigma = 4)
    return patch_deformed


def flipping():
    """
    Flips the image horizontally and/or vertically.
    :param patch: List of 2 or 3 ndarrays [image,mask,(weights)]
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List of flipped ndarrays [flipped_image, flipped_mask]
    """

    vertical_flip = False
    horizontal_flip = False
    
    """
    if len(patch) == 3:
        weights = patch[2]
     """

    # First we toss a coin and depending on the result we flip the image vertically.
    vertical  = np.random.binomial(1, 0.5, 1)
    if vertical == 1 :
        
        vertical_flip = True
        """
        if len(patch) == 3:
            weights = np.fliplr(weights)"""

        print('flipping left-right')
    # Then we toss a coin and depending on the result we flip the image horizontally.

    horizontal = np.random.binomial(1, 0.5, 1)
    if horizontal == 1:
        
        horizontal_flip = True
        
        """
        if len(patch) == 3:
            weights = np.flipud(weights)
        """
        
        
        print('flipping up-down')
    """if len(patch) == 3:
        return [image, mask, weights]
    else:
        return [image, mask]
        """
    return ((vertical_flip, horizontal_flip))




def gaussian_blur(patch, sigma_max=3, verbose=1):
    """
    Adding a gaussian blur to the image.
    :param patch: List of 2 or 3 ndarrays [image,mask,(weights)]
    :param sigma_max: Float, max possible value of the gaussian blur.
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List of 2 or 3 ndarrays [blurred_image, original_mask, (original_weights)]
    """

    image = patch[0]
    mask = patch[1]
    if len(patch) == 3:
        weights = patch[2]
    # Choosing the parameter and applying the transformation
    sigma = np.random.uniform(0,sigma_max, 1)[0]
    if verbose>=1:
        print(('maximum sigma: ', sigma))
    image = gaussian(image, sigma=sigma, preserve_range=True) 

    if len(patch) ==3:
        return [image, mask, weights]
    else:
        return [image, mask]
