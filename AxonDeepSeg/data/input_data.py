from skimage import exposure
from skimage.transform import rescale
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imread
from sklearn import preprocessing
from skimage import transform
from scipy import ndimage
import numpy as np
import random
import os
#import matplotlib.pyplot as plt


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
    shifted_image = img[begin_h:begin_h+256,begin_w:begin_w+256]
    shifted_mask = mask[begin_h:begin_h+256,begin_w:begin_w+256]

    return [shifted_image,shifted_mask]


def rescaling(patch, thresh_indices = [0,0.5]):
    """
    :param patch:  [image,mask]
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: random rescaling of the pair [image,mask]

    --- Rescaling reinforces axons size diversity ---
    """

    scale = random.choice([0.5, 0.75, 1.0, 1.5, 2.0])

    if scale == 1.0:
        rescaled_patch = patch

    else :
        image_rescale = rescale(patch[0], scale, preserve_range= True)
        mask_rescale = rescale(patch[1], scale, preserve_range= True)
        s_r = mask_rescale.shape[0]
        q_h, r_h = divmod(256-s_r,2)

        if q_h > 0:
            image_rescale = np.pad(image_rescale,(q_h, q_h+r_h), mode = "reflect")
            mask_rescale = np.pad(mask_rescale,(q_h, q_h+r_h), mode = "reflect")
        else:           
            patches = extract_patches(image_rescale, mask_rescale, 256)
            i = np.random.randint(len(patches), size=1)[0]
            image_rescale, mask_rescale = patches[i]

        mask_rescale = np.array(mask_rescale)

        for indice,value in enumerate(thresh_indices[:-1]):
            if np.max(mask_rescale) > 1.001:
                thresh_inf = np.int(255*value)
                thresh_sup = np.int(255*thresh_indices[indice+1])
            else:
                thresh_inf = value
                thresh_sup = thresh_indices[indice+1]   

            mask_rescale[(mask_rescale >= thresh_inf) & (mask_rescale < thresh_sup)] = np.mean([value,thresh_indices[indice+1]])

        mask_rescale[(mask_rescale >= thresh_indices[-1])] = 1
        rescaled_patch = [image_rescale.astype(np.uint8), mask_rescale]

    return rescaled_patch


def random_rotation(patch, thresh_indices = [0,0.5]):
    """
    :param patch: [image, mask]
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: random rotation of the pair [image,mask]
    """
    img = patch[0]
    mask = patch[1]

    angle = np.random.uniform(5, 89, 1)

    image_rotated = transform.rotate(img, angle, resize = False, mode = 'symmetric',preserve_range=True)
    gt_rotated = transform.rotate(mask, angle, resize = False, mode = 'symmetric', preserve_range=True)

    for indice,value in enumerate(thresh_indices[:-1]):
        if np.max(gt_rotated) > 1.001:
            thresh_inf = np.int(255*value)
            thresh_sup = np.int(255*thresh_indices[indice+1])
        else:
            thresh_inf = value
            thresh_sup = thresh_indices[indice+1]      
        
        gt_rotated[(gt_rotated >= thresh_inf) & (gt_rotated < thresh_sup)] = np.mean([value,thresh_indices[indice+1]])
    
    gt_rotated[gt_rotated >= thresh_sup] = 1

    return [image_rotated.astype(np.uint8), gt_rotated]


def elastic_transform(image, gt, alpha, sigma, thresh_indices = [0,0.5]):
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

    for indice,value in enumerate(thresh_indices[:-1]):
        if np.max(elastic_gt) > 1.001:
            thresh_inf = np.int(255*value)
            thresh_sup = np.int(255*thresh_indices[indice+1])
            class_max = 255
        else:
            thresh_inf = value
            thresh_sup = thresh_indices[indice+1]  
            class_max = 1
        elastic_gt[(elastic_gt >= thresh_inf) & (elastic_gt < thresh_sup)] = np.mean([value,thresh_indices[indice+1]])

    elastic_gt[elastic_gt >= thresh_sup] = 1

    return [elastic_image, elastic_gt]

def elastic(patch, thresh_indices = [0,0.5]):
    """
    :param patch: [image,mask].
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: random deformation of the pair [image,mask].
    """
    alpha = random.choice([1,2,3,4,5,6,7,8,9])
    patch_deformed = elastic_transform(patch[0],patch[1], alpha = alpha, sigma = 4,thresh_indices = thresh_indices)
    return patch_deformed


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


def random_transformation(patch, thresh_indices = [0,0.5]):
    """
    :param patch: [image,mask].
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: application of the random transformations to the pair [image,mask].
    """
    patch = shifting(patch)
    """plt.figure(1)
    plt.subplot(2,1,1)
    plt.imshow(patch[0],cmap='gray')
    plt.subplot(2,1,2)
    plt.imshow(patch[1],cmap='gray')
    plt.show()"""
    patch = rescaling(patch, thresh_indices = thresh_indices)

    patch = random_rotation(patch, thresh_indices = thresh_indices)  

    patch = elastic(patch, thresh_indices = thresh_indices)  

    patch = flipped(patch)


    return patch


#######################################################################################################################
#                                             Input data for the U-Net                                                #
#######################################################################################################################
class input_data:
    """
    Data to feed the learning/testing of the CNN
    """

    def __init__(self, trainingset_path, type = 'train', thresh_indices = [0,0.5]):
        """
        Input: 
            trainingset_path : string : path to the trainingset folder containing 2 folders Test and Train
                                    with images and ground truthes.
            type : string 'train' or 'test' : for the network's training. 
            thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
        Output:
            None.
        """
        if type == 'train' : # Data for the train
            self.path = trainingset_path+'/Train/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        if type == 'test': # Data for the test
            self.path = trainingset_path+'/Test/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        self.size_image = 256
        self.n_labels = 2
        self.batch_start = 0
        self.thresh_indices = thresh_indices


    def set_batch_start(self, start = 0):
        """
        :param start: starting indice of the data reading by the network.
        :return:
        """
        self.batch_start = start


    def next_batch(self, batch_size = 1, rnd = False, augmented_data = False):
        """
        :param batch_size: number of images per batch to feed the network, 1 image is often enough.
        :param rnd: if True, batch is randomly taken into the training set.
        :param augmented_data: if True, each patch of the batch is randomly transformed with the data augmentation process.
        :return: The pair [batch_x (data), batch_y (prediction)] to feed the network.
        """
        batch_x = []

        # Set the range of indices
        # Read the image and mask files.
        for i in range(batch_size) :
            if rnd :
                indice = random.choice(range(self.set_size))
            else :
                indice = self.batch_start
                self.batch_start += 1
                if self.batch_start >= self.set_size:
                    self.batch_start= 0

            image = imread(self.path + 'image_%s.png' % indice, flatten=False, mode='L')

            mask = imread(self.path + 'mask_%s.png' % indice, flatten=False, mode='L')

            # Online data augmentation
            if augmented_data:
                [image, mask] = random_transformation([image, mask], thresh_indices = self.thresh_indices)
            else:
                for indice,value in enumerate(self.thresh_indices[:-1]):
                    if np.max(mask) > 1.001:
                        thresh_inf = np.int(255*value)
                        thresh_sup = np.int(255*self.thresh_indices[indice+1])
                        class_max = 255
                    else:
                        thresh_inf = value
                        thresh_sup = tself.hresh_indices[indice+1]  
                        class_max = 1
                    mask[(mask >= thresh_inf) & (mask < thresh_sup)] = np.mean([value,self.thresh_indices[indice+1]])

                mask[mask >= thresh_sup] = 1

            #-----PreProcessing --------
            image = exposure.equalize_hist(image) #histogram equalization
            image = (image - np.mean(image))/np.std(image) #data whitening
            #---------------------------
            
            """plt.figure()
            plt.subplot(2,1,1)
            plt.imshow(image,cmap='gray')
            plt.subplot(2,1,2)
            plt.imshow(mask,cmap='gray')
            plt.show()"""

            batch_x.append(image)
            if i == 0:
                batch_y = mask.reshape(-1,1)
            else:
                batch_y = np.concatenate((batch_y, mask.reshape(-1, 1)), axis=0)

        n = len(self.thresh_indices)
        batch_y_tot = np.zeros([batch_y.shape[0], n*batch_y.shape[1]])

        for classe in range(n-1):
            batch_y_tot[:,classe] = (batch_y == np.mean([self.thresh_indices[classe],
                                                             self.thresh_indices[classe+1]]))[:,0]

        batch_y_tot[:,n-1] = (batch_y == 1)[:,0]

        batch_y_tot = batch_y_tot.astype(np.uint8)

        return [np.asarray(batch_x), batch_y_tot]


    def next_batch_WithWeights(self, batch_size = 1, rnd = False, augmented_data = True):
        """
        :param batch_size: number of images per batch to feed the network, 1 image is often enough.
        :param rnd: if True, batch is randomly taken into the training set.
        :param augmented_data: if True, each patch of the batch is randomly transformed with the data augmentation process.
        :return: The triplet [batch_x (data), batch_y (prediction), weights (based on distance to edges)] to feed the network.
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

            image = imread(self.path + 'image_%s.png' % indice, flatten=False, mode='L')
            mask = imread(self.path + 'mask_%s.png' % indice, flatten=False, mode='L')

            if augmented_data:
                [image, mask] = random_transformation([image, mask], thresh_indices = self.thresh_indices)
            else:
                for indice,value in enumerate(self.thresh_indices[:-1]):
                    if np.max(mask) > 1.001:
                        thresh_inf = np.int(255*value)
                        thresh_sup = np.int(255*self.thresh_indices[indice+1])
                        class_max = 255
                    else:
                        thresh_inf = value
                        thresh_sup = tself.hresh_indices[indice+1]  
                        class_max = 1
                    mask[(mask >= thresh_inf) & (mask < thresh_sup)] = np.mean([value,self.thresh_indices[indice+1]])

                mask[mask >= thresh_sup] = 1

            #-----PreProcessing --------
            image = exposure.equalize_hist(image) #histogram equalization
            image = (image - np.mean(image))/np.std(image) #data whitening
            #---------------------------

            to_use = np.asarray(255*mask,dtype='uint8')
            to_use[to_use<=np.min(to_use)]=0
            weight = ndimage.distance_transform_edt(to_use)
            weight[weight==0]=np.max(weight)
            w0 = 10
            sigma = 5

            weight = w0*np.exp(-(weight/sigma)**2/2)
            weight = preprocessing.normalize(weight)

            batch_x.append(image)
            if i == 0:
                batch_y = mask.reshape(-1,1)
                weights = weight.reshape(-1,1)
            else:
                batch_y = np.concatenate((batch_y, mask.reshape(-1, 1)), axis=0)
                weights = np.concatenate((weights, weight.reshape(-1, 1)), axis=0)
        
        n = len(self.thresh_indices)
        batch_y_tot = np.zeros([batch_y.shape[0], n])

        for classe in range(n-1):
            batch_y_tot[:,classe] = (batch_y == np.mean([self.thresh_indices[classe],
                                                             self.thresh_indices[classe+1]]))[:,0]
        batch_y_tot[:,n-1] = (batch_y == 1)[:,0]

        return [np.asarray(batch_x), batch_y_tot.astype(np.uint8), weights]

    def read_batch(self, batch_y, size_batch):
        images = batch_y.reshape(size_batch, self.size_image, self.size_image, self.n_labels)
        return images

