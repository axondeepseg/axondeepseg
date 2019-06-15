import keras
import cv2
import numpy as np
import os


class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=512, thresh_indices=[0, 0.2, 0.8]):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.thresh_indices = thresh_indices

    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, 'image_' + id_name) + ".png"
        mask_path = os.path.join(self.path, 'mask_' + id_name) + ".png"
        # all_masks = os.listdir(mask_path)

        ## Reading Image
        image = cv2.imread(image_path)

        # -----PreProcessing --------
        # image = exposure.equalize_hist(image) #histogram equalization
        # image = (image - np.mean(image))/np.std(image) #data whitening
        # ---------------------------
        # image = np.reshape(image, (self.image_size, self.image_size,1))

        # -----PreProcessing --------
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = descritize_mask(mask, self.thresh_indices)
        # ---------------------------

        # print(mask.shape)
        # mask = np.reshape(mask, (self.image_size, self.image_size,3))
        # mask = np.zeros((self.image_size, self.image_size, 1))
        """
        ## Reading Masks
        for name in all_masks:
            _mask_path = mask_path + name
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size)) #128x128
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image)
        """
        ## Normalizaing
        # image = image/255.0
        # mask = mask/255.0
        # image = image.reshape((512* 512, 3))
        # mask  = mask.reshape((512*512, 3))

        return (image, mask)

    def __getitem__(self, index):

        """
        data_gen_args = dict(vertical_flip=True)
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        seed = 2018


        image_generator = image_datagen.flow(self.__load__(id_name)[0], seed=seed, batch_size= batch_size, shuffle=True)
        mask_generator = mask_datagen.flow(self.__load__(id_name)[1], seed=seed, batch_size= batch_size, shuffle=True)

        """

        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return (image, mask)

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


def labellize_mask_2d(patch, thresh_indices=[0, 0.2, 0.8]):
    '''
    Process a patch with 8 bit pixels ([0-255]) so that the pixels between two threshold values are set to the closest threshold, effectively
    enabling the creation of a mask with as many different values as there are thresholds.

    Returns mask in [0-1] domain
    '''
    mask = np.zeros_like(patch)
    for indice in range(len(thresh_indices) - 1):
        thresh_inf_8bit = 255 * thresh_indices[indice]
        thresh_sup_8bit = 255 * thresh_indices[indice + 1]

        idx = np.where(
            (patch >= thresh_inf_8bit) & (patch < thresh_sup_8bit))  # returns (x, y) of the corresponding indices
        mask[idx] = np.mean([thresh_inf_8bit / 255, thresh_sup_8bit / 255])

    mask[(patch >= 255 * thresh_indices[-1])] = 1

    return patch


def descritize_mask(mask, thresh_indices):
    # Discretization of the mask
    mask = labellize_mask_2d(mask, thresh_indices)  # mask intensity float between 0-1

    # Working out the real mask (sparse cube with n depth layer, one for each class)
    n = len(thresh_indices)  # number of classes
    thresh_indices = [255 * x for x in thresh_indices]
    real_mask = np.zeros([mask.shape[0], mask.shape[1], n])

    for class_ in range(n - 1):
        real_mask[:, :, class_] = (mask[:, :] >= thresh_indices[class_]) * (mask[:, :] < thresh_indices[class_ + 1])
    real_mask[:, :, -1] = (mask[:, :] >= thresh_indices[-1])
    real_mask = real_mask.astype(np.uint8)

    return real_mask