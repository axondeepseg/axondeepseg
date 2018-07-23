import numpy as np
import pandas as pd
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio
import AxonDeepSeg.ads_utils


def get_masks(path_prediction):
    prediction = imageio.imread(path_prediction)

    # compute the axon mask
    axon_prediction = prediction > 200

    # compute the myelin mask
    myelin_prediction = prediction > 100
    myelin_prediction = myelin_prediction ^ axon_prediction

    # get main path
    path_folder, file_name = os.path.split(path_prediction)

    tmp_path = path_prediction.split('_seg-axonmyelin')

    # save the masks
    if tmp_path[0].endswith('.png'):
        tmp_path[0] = os.path.splitext(tmp_path[0])[0]

    imageio.imwrite(tmp_path[0] + '_seg-axon.png', axon_prediction.astype(int))
    imageio.imwrite(tmp_path[0] + '_seg-myelin.png', myelin_prediction.astype(int))

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
