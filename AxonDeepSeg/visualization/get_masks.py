from pathlib import Path

# Scientific modules import
import numpy as np
import pandas as pd
from skimage import io

# AxonDeepSeg modules import
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.ads_utils import convert_path

def get_masks(path_prediction):
    # If string, convert to Path objects
    path_prediction = convert_path(path_prediction)

    prediction = ads.imread(path_prediction)

    # compute the axon mask
    axon_prediction = prediction > 200

    # compute the myelin mask
    myelin_prediction = prediction > 100
    myelin_prediction = myelin_prediction ^ axon_prediction

    # We want to keep the filename path up to the '_seg-axonmyelin' part
    folder_path = path_prediction.parent
    filename_part = path_prediction.name.split('_seg-axonmyelin')[0]
    # Extra check to ensure that the extension was removed
    if filename_part.endswith('.png'):
        filename_part = filename_part.split('.png')[0]

    # Save masks
    filename_axon   = filename_part + '_seg-axon.png'
    filename_myelin = filename_part + '_seg-myelin.png'
    ads.imwrite(folder_path / filename_axon, axon_prediction.astype(int))
    ads.imwrite(folder_path / filename_myelin, myelin_prediction.astype(int))

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
        # If string, convert to Path objects
        writing_path = convert_path(writing_path)
        ads.imwrite(writing_path, rgb_mask)

    return rgb_mask

def get_image_unique_vals_properties(image):
    """
    Returns dict with image unique values properties.
    :param image: np.ndarray or string path to image an file.
    :return: image_properties: dict.
        Keys:
            num_uniques: Integer number of unique pixel values in the image.
            unique_values: Array containing the unique pixel values in the
                           image.
    """
    if not isinstance(image, np.ndarray):
        try:
            image = ads.imread(image)
        except:
            raise IOError('AxonDeepSeg.get_image_unique_vals_properties: Error '
                          'reading image. Function arg must be either an '
                          'np.ndarray or string path to an image file.')

    image_properties = dict()
    image_properties['unique_values'] = np.unique(image)
    image_properties['num_uniques'] = len(image_properties['unique_values'])

    return image_properties
