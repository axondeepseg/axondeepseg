from pathlib import Path

# Scientific modules import
import numpy as np
import pandas as pd
from skimage import io
from scipy.misc import imread, imsave
import imageio

# AxonDeepSeg modules import
import AxonDeepSeg.ads_utils


def get_masks(path_prediction):
    prediction = imageio.imread(path_prediction)

    # compute the axon mask
    axon_prediction = prediction > 200

    # compute the myelin mask
    myelin_prediction = prediction > 100
    myelin_prediction = myelin_prediction ^ axon_prediction

    # We want to keep the filename path up to the '_seg-axonmyelin' part
    path = path_prediction.parent
    broken_filename = path_prediction.name.split('_seg-axonmyelin')[0]
    # Extra check to ensure that the extension was removed
    if broken_filename.endswith('.png'):
        broken_filename = broken_filename.split('.png')[0]

    # Save masks
    imageio.imwrite(path.joinpath(broken_filename + '_seg-axon.png'), axon_prediction.astype(int))
    imageio.imwrite(path.joinpath(broken_filename + '_seg-myelin.png'), myelin_prediction.astype(int))

    return axon_prediction, myelin_prediction


def rgb_rendering_of_mask(pred_img, writing_path=None):
    """
    Returns a segmentation mask in red and blue display
    :param pred_img: segmented image - 3-class mask
    :param save_mask: Boolean: whether or not to save the returned mask
    :param writing_path: string: path where to save the mask if save_mask=True
    :return: rgb_mask: imageio.core.util.Image
    """
    pred_axon = pred_img == 255
    pred_myelin = pred_img == 127

    rgb_mask = np.zeros([np.shape(pred_img)[0], np.shape(pred_img)[1], 3])

    rgb_mask[pred_axon] = [0, 0, 255]
    rgb_mask[pred_myelin] = [255, 0, 0]

    if writing_path is not None:
        imageio.imwrite(writing_path, rgb_mask)

    return rgb_mask
