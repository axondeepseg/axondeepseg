from skimage import exposure
from scipy.misc import imread
from scipy import ndimage
import numpy as np
import random
import os
from data_augmentation import shifting, rescaling, flipping, random_rotation, elastic, noise_addition, noise_multiplication
#import matplotlib.pyplot as plt


def random_transformation(patch, thresh_indices = [0,0.5], data_augmentation=[]):
    """
    :param patch: [image,mask].
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: application of the random transformations to the pair [image,mask].
    """
    if 'shifting' in data_augmentation:
        patch = shifting(patch)
    if 'rescaling' in data_augmentation:   
        patch = rescaling(patch, thresh_indices = thresh_indices)
    if 'random_rotation' in data_augmentation:
        patch = random_rotation(patch, thresh_indices = thresh_indices)
    if 'elastic' in data_augmentation:  
        patch = elastic(patch, thresh_indices = thresh_indices)  
    if 'flipping' in data_augmentation:
        patch = flipping(patch) # used until now, the output is not really realistic.
    if 'noise_addition' in data_augmentation:
        patch = noise_addition(patch)
    if 'noise_multiplication' in data_augmentation:
        patch = noise_multiplication(patch)
       
    return patch

def patch_to_mask(patch, thresh_indices=[0, 0.5]):
    '''
    Process a patch so that the pixels between two threshold values are set to the closest threshold, effectively
    enabling the creation of a mask with as many different values as there are thresholds.
    '''

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


    def next_batch(self, batch_size = 1, rnd = False, data_augmentation=[]):
        """
        :param batch_size: number of images per batch to feed the network, 1 image is often enough.
        :param rnd: if True, batch is randomly taken into the training set.
        :param augmented_data: if True, each patch of the batch is randomly transformed with the data augmentation process.
        :return: The pair [batch_x (data), batch_y (prediction)] to feed the network.
        """
                
        batch_x = []
        batch_y = []

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

            # We are reading directly the images. Range of values : 0-255
            image = imread(self.path + 'image_%s.png' % indice, flatten=False, mode='L')
            mask = imread(self.path + 'mask_%s.png' % indice, flatten=False, mode='L')            
            
            # Online data augmentation
            if data_augmentation:
                [image, mask] = random_transformation([image, mask], thresh_indices = self.thresh_indices, data_augmentation) 
            mask = patch_to_mask(mask, self.thresh_indices)
            
                
            #-----PreProcessing --------
            image = exposure.equalize_hist(image) #histogram equalization
            image = (image - np.mean(image))/np.std(image) #data whitening
            #---------------------------
            
            n = len(self.thresh_indices)

            # Working out the real mask (sparse cube with n depth layer for each class)
            real_mask = np.zeros([mask.shape[0], mask.shape[1], n])
            for class_ in range(n-1):
                real_mask[:,:,class_] = (mask[:,:] >= self.thresh_indices[class_]) * (mask[:,:] <                                    self.thresh_indices[class_+1])
            real_mask[:,:,n-1] = (mask > self.thresh_indices[n-1])
            real_mask = real_mask.astype(np.uint8)

            batch_x.append(image)
            batch_y.append(real_mask)
        
        # Ensuring that we do have np.arrays of the good size for batch_x and batch_y before returning them
        return [np.stack(batch_x), np.stack(batch_y)]


    def next_batch_WithWeights(self, batch_size = 1, rnd = False, data_augmentation=[]):
        """
        :param batch_size: number of images per batch to feed the network, 1 image is often enough.
        :param rnd: if True, batch is randomly taken into the training set.
        :param augmented_data: if True, each patch of the batch is randomly transformed with the data augmentation process.
        :return: The triplet [batch_x (data), batch_y (prediction), weights (based on distance to edges)] to feed the network.
        """
        batch_x = []
        batch_y = []
        batch_w = []
        
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
            if data_augmentation:
                [image, mask] = random_transformation([image, mask], thresh_indices = self.thresh_indices, data_augmentation)
            mask = patch_to_mask(mask, self.thresh_indices)

            #-----PreProcessing --------
            image = exposure.equalize_hist(image) #histogram equalization
            image = (image - np.mean(image))/np.std(image) #data whitening
            #---------------------------

            # Create a weight map for each class (background is the first class, equal to 1
            
            
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
            
            # Generating the mask with the real labels as well as the matrix of the weights
            
            n = len(self.thresh_indices) #number of classes

            weights_intermediate = np.reshape(weights_intermediate,[mask.shape[0], mask.shape[1], n])
            
            # Working out the real mask (sparse cube with n depth layer for each class)
            real_mask = np.zeros([mask.shape[0], mask.shape[1], n])
            for class_ in range(n-1):
                real_mask[:,:,class_] = (mask[:,:] >= self.thresh_indices[class_]) * (mask[:,:] <                                    self.thresh_indices[class_+1])
            real_mask[:,:,n-1] = (mask > self.thresh_indices[n-1])
            real_mask = real_mask.astype(np.uint8)
            
            # Working out the real weights (sparse matrix with the weights associated with each pixel)
            
            real_weights = np.zeros([mask.shape[0], mask.shape[1]])            
            for class_ in range(n):
                real_weights += np.multiply(real_mask[:,:,class_],weights_intermediate[:,:,class_])
                
            
            # We have now loaded the good image, a mask (under the shape of a matrix, with different labels) that still needs to be converted to a volume (meaning, a sparse cube where each layer of depth relates to a class)
            
            batch_x.append(image)
            batch_y.append(real_mask)
            batch_w.append(real_weights)

        # We then stack the matrices we generated and return them
        return [np.stack(batch_x), np.stack(batch_y), np.stack(batch_w)]

    def read_batch(self, batch_y, size_batch):
        images = batch_y.reshape(size_batch, self.size_image, self.size_image, self.n_labels)
        return images
    
    