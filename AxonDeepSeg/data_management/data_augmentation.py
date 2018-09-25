from skimage.transform import rescale
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
from skimage.filters import gaussian
import numpy as np
import random
from .patch_extraction import extract_patch
import AxonDeepSeg.ads_utils

#######################################################################################################################
#                                                Data Augmentation                                                    #
#######################################################################################################################

# We find here the functions that perform transformations to images and mask. 
# All functions take 8-bit images and return 8-bit images

def shifting(patch, percentage_max = 0.1, verbose=0):
    """
    Shifts the input by a random number of pixels.
    :param patch: List of 2 or 3 ndarrays, [image,mask, (weights)]
    :param percentage_max: Float, maximum value of the shift, in terms of percentage wrt the size of the input.
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List composed of 2 or 3 random shifted the pair [image,mask, (weights)]
    """

    patch_size = patch[0].shape[0]
    n_classes = patch[1].shape[-1]
    size_shift = int(percentage_max*patch_size) # Maximum shift in pixels.
    image = np.pad(patch[0],size_shift, mode = "reflect")
    mask = np.stack([np.pad(np.squeeze(e),size_shift, mode = "reflect") for e in np.split(patch[1], n_classes, axis=-1)], axis=-1)
    if len(patch) == 3:
        weights = np.pad(patch[2],size_shift, mode = "reflect")

    # Choosing randomly the number of pixels for height and width to shift the images.
    begin_h = np.random.randint(2*size_shift-1)
    begin_w = np.random.randint(2*size_shift-1)
    
    if verbose >= 1:
        print(('height shift: ',begin_h, ', width shift: ', begin_w))     
    
    shifted_image = image[begin_h:begin_h+patch_size,begin_w:begin_w+patch_size]
    shifted_mask = np.stack([np.squeeze(e)[begin_h:begin_h+patch_size,begin_w:begin_w+patch_size] for e in np.split(mask, n_classes, axis=-1)], axis=-1)

    if len(patch) == 3:
        shifted_weights = weights[begin_h:begin_h+patch_size,begin_w:begin_w+patch_size]
        return [shifted_image,shifted_mask, shifted_weights]
    else:
        return [shifted_image, shifted_mask]


def rescaling(patch, factor_max=1.2, verbose=0):
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
    n_classes = patch[1].shape[-1]

    # Randomly choosing the resampling factor.
    scale = np.random.uniform(low_bound, high_bound, 1)[0]
    if verbose >= 1:
        print(('rescaling factor: ', scale))
        
    patch_size = patch[0].shape[0]
    new_patch_size = int(patch_size*scale)

    # If the resampling factor is too close to 1 we do not resample.
    if (new_patch_size <= patch_size+5) and (new_patch_size >= patch_size-5): # To avoid having q_h = 0
        return patch
    else :
        image_rescale = rescale(patch[0], scale, preserve_range= True)
        mask_rescale = rescale(patch[1], scale, preserve_range= True)
        if len(patch) == 3:
            weights_rescale = rescale(patch[2], scale, preserve_range=True)

        s_r = mask_rescale.shape[0]
        q_h, r_h = divmod(patch_size-s_r,2)

        # If we undersample, we pad the rest of the image.
        if q_h > 0:
            image_rescale = np.pad(image_rescale,(q_h, q_h+r_h), mode = "reflect")
            mask_rescale = [np.pad(np.squeeze(e),(q_h, q_h+r_h), mode = "reflect") for e in np.split(mask_rescale, n_classes, axis=-1)]
            mask_rescale = np.stack(mask_rescale, axis=-1)
            weights_rescale = np.pad(weights_rescale,(q_h, q_h+r_h), mode = "reflect")

        # if we oversample
        else:
            to_extract = [image_rescale, mask_rescale]
            if len(patch) == 3:
                to_extract += [weights_rescale]

            # We extract all the patches coming from the oversampled image.
            patches = extract_patch(to_extract, patch_size)
            i = np.random.randint(len(patches), size=1)[0]

            if len(patch) == 3:
                image_rescale, mask_rescale, weights_rescale = patches[i]
            else:
                image_rescale, mask_rescale = patches[i]

        mask_rescale = np.array(mask_rescale)
        if len(patch) == 3:
            weights_rescale = np.array(weights_rescale)
            return [image_rescale.astype(np.uint8), mask_rescale.astype(np.uint8),
                              weights_rescale.astype(np.float32)]
        else:
            return [image_rescale.astype(np.uint8), mask_rescale.astype(np.uint8)]


def random_rotation(patch, low_bound=5, high_bound=89, verbose=0):
    """
    Rotates randomly the input, angle between low_bound and high_bound.
    :param patch: List of 2 or 3 inputs (ndarrays) [image, mask, (weights)]
    :param low_bound: Int, lower bound of the randomly selected rotation angle.
    :param high_bound: Int, higher bound of the randomly selected rotation angle.
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List of 2 or 3 randomly rotated inputs [image,mask, (weights)]
    """

    image = patch[0]
    mask = patch[1]
    if len(patch) == 3:
        weights = patch[2]

    # Randomly choosing the angle of rotation.
    angle = np.random.uniform(low_bound, high_bound, 1)
    
    if verbose >= 1:
        print(('rotation angle: ', angle))

    image_rotated = transform.rotate(image, angle, resize = False, mode = 'symmetric',preserve_range=True)
    mask_rotated = transform.rotate(mask, angle, resize = False, mode = 'symmetric', preserve_range=True)
    
    if len(patch) == 3:
        weights_rotated = transform.rotate(weights, angle, resize=False, mode='symmetric', preserve_range=True)
        return [image_rotated.astype(np.uint8), mask_rotated.astype(np.uint8), weights_rotated.astype(np.float32)]
    else:
        return [image_rotated.astype(np.uint8), mask_rotated.astype(np.uint8)]


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


def flipping(patch, verbose=0):
    """
    Flips the image horizontally and/or vertically.
    :param patch: List of 2 or 3 ndarrays [image,mask,(weights)]
    :param verbose: Int. The higher, the more information is displayed about the transformation.
    :return: List of flipped ndarrays [flipped_image, flipped_mask]
    """

    image = patch[0]
    mask = patch[1]
    if len(patch) == 3:
        weights = patch[2]

    # First we toss a coin and depending on the result we flip the image vertically.
    s = np.random.binomial(1, 0.5, 1)
    if s == 1 :
        image, mask = [np.fliplr(image), np.fliplr(mask)]
        if len(patch) == 3:
            weights = np.fliplr(weights)
        if verbose >= 1:
            print('flipping left-right')
    # Then we toss a coin and depending on the result we flip the image horizontally.

    s = np.random.binomial(1, 0.5, 1)
    if s == 1:
        image, mask = [np.flipud(image), np.flipud(mask)]
        if len(patch) == 3:
            weights = np.flipud(weights)
        if verbose >= 1:
            print('flipping up-down')
    if len(patch) == 3:
        return [image, mask, weights]
    else:
        return [image, mask]



def gaussian_blur(patch, sigma_max=3, verbose=0):
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
