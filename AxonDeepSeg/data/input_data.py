from skimage import exposure
from scipy.misc import imread
from scipy import ndimage
import numpy as np
import random
import os
from data_augmentation import shifting, rescaling, flipping, random_rotation, elastic
#import matplotlib.pyplot as plt


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
    patch = flipping(patch) # used until now, the output is not really realistic.

    for indice,value in enumerate(thresh_indices[:-1]):
        if np.max(patch[1]) > 1.001:
            thresh_inf = np.int(255*value)
            thresh_sup = np.int(255*thresh_indices[indice+1])
        else:
            thresh_inf = value
            thresh_sup = thresh_indices[indice+1]   

        patch[1][(patch[1] >= thresh_inf) & (patch[1] < thresh_sup)] = np.mean([value,thresh_indices[indice+1]])

    patch[1][(patch[1] >= thresh_indices[-1])] = 1

    return patch


#######################################################################################################################
#                                             Input data for the U-Net                                                #
#######################################################################################################################
class input_data:
    """
    Data to feed the learning/validating of the CNN
    """

    def __init__(self, trainingset_path, type = 'train', thresh_indices = [0,0.5], image_size = 256):
        """
        Input: 
            trainingset_path : string : path to the trainingset folder containing 2 folders Validation and Train
                                    with images and ground truthes.
            type : string 'train' or 'validation' : for the network's training.
            thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
        Output:
            None.
        """
        if type == 'train' : # Data for the train !!!!!!!!! change to Training
            self.path = trainingset_path+'/Train/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        if type == 'validation': # Data for the validation
            self.path = trainingset_path+'/Validation/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        self.size_image = image_size
        self.n_labels = 2
        self.batch_start = 0
        self.thresh_indices = thresh_indices

    def get_size(self):
        return self.set_size


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
                    else:
                        thresh_inf = value
                        thresh_sup = self.thresh_indices[indice+1]
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

        for class_ in range(n-1):
            batch_y_tot[:,class_] = (batch_y == np.mean([self.thresh_indices[class_],
                                                             self.thresh_indices[class_+1]]))[:,0]

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

            #Data augmentation. If not, set the mask's values to the labels values.
            if augmented_data:
                [image, mask] = random_transformation([image, mask], thresh_indices = self.thresh_indices)
            else:
                for indice,value in enumerate(self.thresh_indices[:-1]):
                    if np.max(mask) > 1.001:
                        thresh_inf = np.int(255*value)
                        thresh_sup = np.int(255*self.thresh_indices[indice+1])
                    else:
                        thresh_inf = value
                        thresh_sup = self.thresh_indices[indice+1]
                    mask[(mask >= thresh_inf) & (mask < thresh_sup)] = np.mean([value,self.thresh_indices[indice+1]])

                mask[mask >= thresh_sup] = 1

            #-----PreProcessing --------
            image = exposure.equalize_hist(image) #histogram equalization
            image = (image - np.mean(image))/np.std(image) #data whitening
            #---------------------------

            # Create a weight map for each class (background is the first class, equal to 1.
            weights_intermediate = np.ones((self.size_image * self.size_image,len(self.thresh_indices)))
            #weights_intermediate = np.zeros((self.size_image, self.size_image, len(self.thresh_indices[1:])))

            for indice,classe in enumerate(self.thresh_indices[1:]):

                mask_classe = np.asarray(list(mask))
                if classe!=self.thresh_indices[-1]:
                    mask_classe[mask_classe != np.mean([self.thresh_indices[indice - 1], classe])] = 0
                    mask_classe[mask_classe==np.mean([self.thresh_indices[indice-1],classe])]=1
                else:
                    mask_classe[mask_classe!=1]=0

                to_use = np.asarray(255*mask_classe,dtype='uint8')
                to_use[to_use <= np.min(to_use)] = 0
                weight = ndimage.distance_transform_edt(to_use)
                weight[weight==0]=np.max(weight)

                if classe == self.thresh_indices[1]:
                    w0 = 0.5
                else :
                    w0 = 1

                sigma = 2
                weight = 1 + w0*np.exp(-(weight/sigma)**2/2)
                #weight = weight/np.max(weight)
                weights_intermediate[:,indice] = weight.reshape(-1, 1)[:,0]
                #weights_intermediate[:, :, indice] = weight

                """plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(mask,cmap='gray')
                plt.title('Ground truth')
                plt.subplot(2,2,2)
                plt.imshow(weight, interpolation='nearest', cmap='gray',vmin=1)
                plt.title('Weight map')
                plt.colorbar(ticks=[1, 10])
                plt.show()"""

            #weights_intermediate = np.ones((self.size_image * self.size_image, len(self.thresh_indices)))
            #weights_mean = np.mean(weights_intermediate, axis = 2)

            batch_x.append(image)
            if i == 0:
                batch_y = mask.reshape(-1,1)
                weights = weights_intermediate
            else:
                batch_y = np.concatenate((batch_y, mask.reshape(-1, 1)), axis=0)
                #weights = np.concatenate((weights, weights_mean.reshape(-1, 1)), axis=2)
                weights = np.concatenate((weights, weights_intermediate), axis=2)
        
        n = len(self.thresh_indices)
        batch_y_tot = np.zeros([batch_y.shape[0], n*batch_y.shape[1]])

        for class_ in range(n-1):
            batch_y_tot[:,class_] = (batch_y == np.mean([self.thresh_indices[class_],
                                                             self.thresh_indices[class_+1]]))[:,0]
        batch_y_tot[:,n-1] = (batch_y == 1)[:,0]
        batch_y_tot = batch_y_tot.astype(np.uint8)

        return [np.asarray(batch_x), batch_y_tot, weights]

    def read_batch(self, batch_y, size_batch):
        images = batch_y.reshape(size_batch, self.size_image, self.size_image, self.n_labels)
        return images

