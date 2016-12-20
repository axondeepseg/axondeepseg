from skimage import exposure
from skimage.transform import rescale
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imread
from sklearn import preprocessing
from skimage import transform
import numpy as np
import random
import os

def extract_patches(img, mask, size):
    """
    :param img: image represented by a numpy-array
    :param mask: groundtruth of the segmentation
    :param size: size of the patches to extract
    :return: a list of pairs [patch, ground_truth] with a very low overlapping.
    """

    h, w = img.shape

    q_h, r_h = divmod(h, size)
    q_w, r_w = divmod(w, size)

    r2_h = size-r_h
    r2_w = size-r_w

    q3_h, r3_h = divmod(r2_h, q_h)
    q3_w, r3_w = divmod(r2_w, q_w)

    patches = []
    pos_x = 0
    while pos_x+size<=h:
        pos_y = 0
        while pos_y+size<=w:
            patch = img[pos_x:pos_x+size, pos_y:pos_y+size]
            patch_gt = mask[pos_x:pos_x+size, pos_y:pos_y+size]
            patches.append([patch,patch_gt])
            pos_y = size + pos_y - q3_w
            if pos_y + size > w :
                pos_y = pos_y - r3_w

        pos_x = size + pos_x - q3_h
        if pos_x + size > h:
            pos_x = pos_x - r3_h
    return patches

#######################################################################################################################
#                                                Data Augmentation                                                    #
#######################################################################################################################

def flipped(patch):
    """
    :param patch: [image,mask]
    :return: random vertical and horizontal flipped [image,mask]
    """
    s = np.random.binomial(1, 0.5, 1)
    image = patch[0]
    gt = patch[1]
    if s == 1 :
        image, gt = [np.fliplr(image), np.fliplr(gt)]
    s = np.random.binomial(1, 0.5, 1)
    if s == 1:
        image, gt = [np.flipud(image), np.flipud(gt)]
    return [image, gt]


def elastic_transform(image, gt, alpha, sigma):
    """
    :param image: image
    :param gt: ground truth
    :param alpha: deformation coefficient (high alpha -> strong deformation)
    :param sigma: std of the gaussian filter. (high sigma -> smooth deformation)
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
    elastic_gt = preprocessing.binarize(np.array(elastic_gt), threshold=0.5)

    return [elastic_image, elastic_gt]

def elastic(patch):
    """
    :param patch: [image,mask]
    :return: random deformation of the pair [image,mask]
    """
    alpha = random.choice([1,2,3,4,5,6,7,8,9])
    patch_deformed = elastic_transform(patch[0],patch[1], alpha = alpha, sigma = 4)
    return patch_deformed


def shifting(patch):
    """
    :param patch: [image,mask]
    :return: random shifting of the pair [image,mask]
    """
    size_shift = 10
    img = np.pad(patch[0],size_shift, mode = "reflect")
    mask = np.pad(patch[1],size_shift, mode = "reflect")
    begin_h = np.random.randint(2*size_shift-1)
    begin_w = np.random.randint(2*size_shift-1)
    shifted_image = img[begin_h:,begin_w:]
    shifted_mask = mask[begin_h:,begin_w:]

    return [shifted_image,shifted_mask]



def rescaling(patch):
    """
    :param patch:  [image,mask]
    :return: random rescaling of the pair [image,mask]

    --- Rescaling reinforces axons size diversity ---
    """

    scale = random.choice([0.5, 0.75, 1.0, 1.5, 2.0])

    image_rescale = rescale(patch[0], scale)
    mask_rescale = rescale(patch[1], scale)
    s_r = mask_rescale.shape[0]
    q_h, r_h = divmod(256-s_r,2)

    if q_h > 0 :
        image_rescale = np.pad(image_rescale,(q_h, q_h+r_h), mode = "reflect")
        mask_rescale = np.pad(mask_rescale,(q_h, q_h+r_h), mode = "reflect")
    else :
        patches = extract_patches(image_rescale, mask_rescale, 256)
        i = np.random.randint(len(patches), size=1)[0]
        image_rescale,mask_rescale = patches[i]

    mask_rescale = preprocessing.binarize(np.array(mask_rescale), threshold=0.001)
    rescaled_patch = [image_rescale, mask_rescale]

    return rescaled_patch

def random_rotation(patch):
    """
    :param patch: [image, mask]
    :return: random rotation of the pair [image,mask]
    """

    img = np.pad(patch[0],180,mode = "reflect")
    mask = np.pad(patch[1],180,mode = "reflect")

    angle = np.random.uniform(5, 89, 1)

    image_size = (img.shape[1], img.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    image_rotated = transform.rotate(img, angle, resize=True, center=image_center, preserve_range=True).astype(int)
    gt_rotated = transform.rotate(mask, angle, resize=True, center=image_center, preserve_range=True)
    gt_rotated = (preprocessing.binarize(gt_rotated, threshold=0.5)).astype(int)

    s_p = image_rotated.shape[0]
    center = int(float(s_p)/2)

    image_rotated_cropped = image_rotated[center-128:center+128, center-128:center+128]
    gt_rotated_cropped = gt_rotated[center-128:center+128, center-128:center+128]

    return [image_rotated_cropped, gt_rotated_cropped]


def random_transformation(patch):
    """
    :param patch: [image,mask]
    :param size: application of the random transformations to the pair [image,mask]
    :return: application of the random transformations to the pair [image,mask]
    """
    patch = shifting(patch)
    patch = rescaling(patch)
    patch = random_rotation(patch)
    patch = elastic(patch)
    patch = flipped(patch)
    return patch


#######################################################################################################################
#                                             Input data for the U-Net                                                #
#######################################################################################################################
class input_data:
    """
    Data to feed the learning/testing of the CNN
    """

    def __init__(self, trainingset_path, type = 'train'):

        if type == 'train' : # Data for the train
            self.path = trainingset_path+'/Train/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        if type == 'test': # Data for the test
            self.path = trainingset_path+'/Test/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        self.size_image = 256
        self.n_labels = 2
        self.batch_start = 0

    def set_batch_start(self, start = 0):
        """
        :param start: starting indice of the data reading by the network
        :return:
        """
        self.batch_start = start


    def next_batch(self, batch_size = 1, rnd = False, augmented_data = True):
        """
        :param batch_size: number of images per batch to feed the network, 1 image is often enough
        :param rnd: if True, batch is randomly taken into the training set
        :param augmented_data: if True, each patch of the batch is randomly transformed with the data augmentation process
        :return: The pair [batch_x (data), batch_y (prediction)] to feed the network
        """
        batch_x = []
        for i in range(batch_size) :
            if rnd :
                indice = random.choice(range(self.set_size))
            else :
                indice = self.batch_start
                self.batch_start += 1
                if self.batch_start >= self.set_size:
                    self.batch_start= 0

            image = imread(self.path + 'image_%s.jpeg' % indice, flatten=False, mode='L')

            #-----PreProcessing --------
            image = exposure.equalize_hist(image) #histogram equalization
            image = (image - np.mean(image))/np.std(image) #data whitening
            #---------------------------
            mask = preprocessing.binarize(imread(self.path + 'mask_%s.jpeg' % indice, flatten=False, mode='L'), threshold=125)

            if augmented_data :
                [image, mask] = random_transformation([image, mask])

            batch_x.append(image)
            if i == 0:
                batch_y = mask.reshape(-1,1)
            else:
                batch_y = np.concatenate((batch_y, mask.reshape(-1, 1)), axis=0)
        batch_y = np.concatenate((np.invert(batch_y)/255, batch_y), axis = 1)

        return [np.asarray(batch_x), batch_y]

    def read_batch(self, batch_y, size_batch):
        images = batch_y.reshape(size_batch, self.size_image, self.size_image, self.n_labels)
        return images

