import keras
import numpy as np
import AxonDeepSeg.ads_utils as ads

from scipy import ndimage
from skimage import exposure

import AxonDeepSeg.ads_utils

from AxonDeepSeg.ads_utils import convert_path


class DataGen(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
        self,
        ids,
        path,
        augmentations,
        batch_size=8,
        image_size=512,
        thresh_indices=[0, 0.2, 0.8],
    ):
        """
          Initalization for the DataGen class
          :param ids: List of strings, ids of all the images/masks in the training set.
          :param batch_size: Int, the batch size used for training.
          :param image_size: Int, input image size.
          :param image_size: Int, input image size.
          :param augmentations: Compose object, a set of data augmentation operations to be applied.
          :return: the original image, a list of patches, and their positions.
        """

        # If string, convert to Path object
        path = convert_path(path)

        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.thresh_indices = thresh_indices
        self.augment = augmentations

    def __load__(self, id_name):
        """
          Loads images and masks
          :param ids_name: String, id name of a particular image/mask.
        """

        ## Path
        image_path = self.path / ("image_" + id_name + ".png")
        mask_path = self.path / ("mask_" + id_name + ".png")
        ## Reading Image
        image = ads.imread(str(image_path))
        image = np.reshape(image, (self.image_size, self.image_size, 1))

        # -----Mask PreProcessing --------
        mask = ads.imread(str(mask_path))
        mask = descritize_mask(mask, self.thresh_indices)
        # ---------------------------
        return (image, mask)

    def __getitem__(self, index):
        """
          Generates a batch of  images/masks
          :param ids_name: String, id name of a particular image/mask..
        """
        files_batch = self.ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        image = []
        mask = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        images = np.array(image)
        masks = np.array(mask)

        image_aug = []
        mask_aug = []
        for x, y in zip(images, masks):
            aug = self.augment(image=x, mask=y)
            image_aug.append(aug["image"])
            mask_aug.append(aug["mask"])
        image_aug = np.array(image_aug)
        mask_aug = np.array(mask_aug)
        return (image_aug, mask_aug)

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


def labellize_mask_2d(patch, thresh_indices=[0, 0.2, 0.8]):
    """
    Process a patch with 8 bit pixels ([0-255]) so that the pixels between two threshold values are set to the closest threshold, effectively
    enabling the creation of a mask with as many different values as there are thresholds.

    Returns mask in [0-1] domain
    """
    mask = np.zeros_like(patch)
    for indice in range(len(thresh_indices) - 1):
        thresh_inf_8bit = 255 * thresh_indices[indice]
        thresh_sup_8bit = 255 * thresh_indices[indice + 1]

        idx = np.where(
            (patch >= thresh_inf_8bit) & (patch < thresh_sup_8bit)
        )  # returns (x, y) of the corresponding indices
        mask[idx] = np.mean([thresh_inf_8bit / 255, thresh_sup_8bit / 255])

    mask[(patch >= 255 * thresh_indices[-1])] = 1

    return patch


def descritize_mask(mask, thresh_indices):
    """
        Process a mask with 8 bit pixels ([0-255]) such that it get discretizes into 3 different channels ( background, myelin, axon) .
        Returns mask composed of 3 different channels ( background, myelin, axon )
    """

    # Discretization of the mask
    mask = labellize_mask_2d(
        mask, thresh_indices
    )  # mask intensity float between 0-1

    # Working out the real mask (sparse cube with n depth layer, one for each class)
    n = len(thresh_indices)  # number of classes
    thresh_indices = [255 * x for x in thresh_indices]
    real_mask = np.zeros([mask.shape[0], mask.shape[1], n])

    for class_ in range(n - 1):
        real_mask[:, :, class_] = (mask[:, :] >= thresh_indices[class_]) * (
            mask[:, :] < thresh_indices[class_ + 1]
        )
    real_mask[:, :, -1] = mask[:, :] >= thresh_indices[-1]
    real_mask = real_mask.astype(np.uint8)

    return real_mask
