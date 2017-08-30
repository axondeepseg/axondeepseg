from skimage import exposure
from skimage.transform import rescale
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imread
from sklearn import preprocessing
from skimage import transform
from skimage.filters import gaussian
from scipy import ndimage
import numpy as np
import random
import os
from patch_extraction import extract_patch

#######################################################################################################################
#                                                Data Augmentation                                                    #
#######################################################################################################################

# We find here the functions that perform transformations to images and mask. 
# All functions take 8-bit images and return 8-bit images

def shifting(patch, percentage_max = 0.1, verbose=0):
    """
    :param patch: [image,mask]
    :return: random shifting of the pair [image,mask]
    """

    patch_size = patch[0].shape[0]
    size_shift = int(percentage_max*patch_size)
    img = np.pad(patch[0],size_shift, mode = "reflect")
    mask = np.pad(patch[1],size_shift, mode = "reflect")
    begin_h = np.random.randint(2*size_shift-1)
    begin_w = np.random.randint(2*size_shift-1)
    
    if verbose >= 1:
        print 'height shift: ',begin_h, ', width shift: ', begin_w     
    
    shifted_image = img[begin_h:begin_h+patch_size,begin_w:begin_w+patch_size]
    shifted_mask = mask[begin_h:begin_h+patch_size,begin_w:begin_w+patch_size]

    return [shifted_image,shifted_mask]


def rescaling(patch, factor_max=1.2, verbose=0): #indices to indexes.
    """
    :param patch:  [image,mask]
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: random rescaling of the pair [image,mask]

    --- Rescaling reinforces axons size diversity ---
    """
    
    low_bound = 1.0*factor_max
    high_bound = 1.0/factor_max

    scale = np.random.uniform(low_bound, high_bound, 1)[0]
    if verbose >= 1:
        print 'rescaling factor: ', scale
        
    patch_size = patch[0].shape[0]
    
    if scale == 1.0:
        rescaled_patch = patch
    else :
        image_rescale = rescale(patch[0], scale, preserve_range= True)
        mask_rescale = rescale(patch[1], scale, preserve_range= True)
        s_r = mask_rescale.shape[0]
        q_h, r_h = divmod(patch_size-s_r,2)

        if q_h > 0:
            image_rescale = np.pad(image_rescale,(q_h, q_h+r_h), mode = "reflect")
            mask_rescale = np.pad(mask_rescale,(q_h, q_h+r_h), mode = "reflect")

        else:           
            patches = extract_patch(image_rescale, mask_rescale, patch_size)
            i = np.random.randint(len(patches), size=1)[0]
            image_rescale, mask_rescale = patches[i]


        mask_rescale = np.array(mask_rescale)
        rescaled_patch = [image_rescale.astype(np.uint8), mask_rescale.astype(np.uint8)]
    return rescaled_patch


def random_rotation(patch, low_bound=5, high_bound=89, verbose=0):
    """
    :param patch: [image, mask]
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: random rotation of the pair [image,mask]
    """
    img = patch[0]
    mask = patch[1]

    angle = np.random.uniform(low_bound, high_bound, 1)
    
    if verbose >= 1:
        print 'rotation angle: ', angle

    image_rotated = transform.rotate(img, angle, resize = False, mode = 'symmetric',preserve_range=True)
    gt_rotated = transform.rotate(mask, angle, resize = False, mode = 'symmetric', preserve_range=True)

    return [image_rotated.astype(np.uint8), gt_rotated.astype(np.uint8)]


def elastic_transform(image, gt, alpha, sigma):
    """
    :param image: image
    :param gt: ground truth
    :param alpha: deformation coefficient (high alpha -> strong deformation)
    :param sigma: std of the gaussian filter. (high sigma -> smooth deformation)
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: deformation of the pair [image,mask]
    """

    random_state = np.random.RandomState(None)
    shape = image.shape

    d = 4
    sub_shape = (shape[0]/d, shape[0]/d)

    deformations_x = random_state.rand(*sub_shape) * 2 - 1
    deformations_y = random_state.rand(*sub_shape) * 2 - 1

    deformations_x = np.repeat(np.repeat(deformations_x, d, axis=1), d, axis = 0)
    deformations_y = np.repeat(np.repeat(deformations_y, d, axis=1), d, axis = 0)

    dx = gaussian_filter(deformations_x, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(deformations_y, sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    elastic_image = map_coordinates(image, indices, order=1).reshape(shape)
    elastic_gt = map_coordinates(gt, indices, order=1).reshape(shape)
    elastic_gt = np.array(elastic_gt)


    return [elastic_image.astype(np.uint8), elastic_gt.astype(np.uint8)]

def elastic(patch, alpha_max=9, verbose=0):
    """
    :param patch: [image,mask].
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: random deformation of the pair [image,mask].
    """
    alpha = random.choice(range(1, alpha_max))
    
    if verbose>=1:
        print 'elastic transform alpha coeff: ', alpha
    
    patch_deformed = elastic_transform(patch[0],patch[1], alpha = alpha, sigma = 4)
    return patch_deformed


def flipping(patch, verbose=0):
    """
    :param patch: [image,mask]
    :return: random vertical and horizontal flipped [image,mask]
    """
    s = np.random.binomial(1, 0.5, 1)
    image = patch[0]
    gt = patch[1]
    if s == 1 :
        image, gt = [np.fliplr(image), np.fliplr(gt)]
        if verbose >= 1:
            print 'flipping left-right'
    s = np.random.binomial(1, 0.5, 1)
    if s == 1:
        image, gt = [np.flipud(image), np.flipud(gt)]
        if verbose >= 1:
            print 'flipping up-down'
    return [image, gt]


def gaussian_blur(patch, sigma_max=3, verbose=0):
    '''
    :param patch: [image, mask]
    :return image and patch where a gaussian filter was applied to the image (adds blur).
    '''
    
    image, gt = patch
    # Choosing the parameter and applying the transformation
    sigma = np.random.uniform(0,sigma_max, 1)[0]
    if verbose>=1:
        print 'maximum sigma: ', sigma
    image = gaussian(image, sigma=sigma, preserve_range=True) 
    
    return [image, gt]
















