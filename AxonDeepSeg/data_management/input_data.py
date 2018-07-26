from skimage import exposure
from scipy.misc import imread
from scipy import ndimage
import numpy as np
import random
import os
from .data_augmentation import *
from AxonDeepSeg.patch_management_tools import apply_legacy_preprocess, apply_preprocess
import functools
import copy
import AxonDeepSeg.ads_utils


def generate_list_transformations(transformations = {}, thresh_indices = [0,0.5], verbose=0):
    
    L_transformations = []

    dict_transformations = {'shifting':shifting,
                            'rescaling':rescaling,
                            'random_rotation':random_rotation,
                            'elastic':elastic,
                            'flipping':flipping,
                            'gaussian_blur':gaussian_blur
                           }
    
    if transformations == {}:
        L_transformations = [functools.partial(v, verbose=verbose) for k,v in list(dict_transformations.items())]
    else:
        #print(transformations)
        L_c = []
        for k,v in list(transformations.items()):
            if (k != 'type') and (v['activate']==True):
                number = v['order']
                c = (number,k,v)
                L_c.append(c)
        # We sort the transformations to make by the number preceding the transformation in the dict in the config file        
        L_c_sorted = sorted(L_c, key=lambda x: x[0]) 
        
        # Creation of the list of transformations to apply
        for tup in L_c_sorted:
            k = tup[1]
            v = tup[2]
            list(map(v.pop, ['order','activate']))
            L_transformations.append(functools.partial(dict_transformations[k], verbose=verbose, **v))
            
    return L_transformations


def all_transformations(patch, thresh_indices = [0,0.5], transformations = {}, verbose=0):
    """
    :param patch: [image,mask].
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: application of the random transformations to the pair [image,mask].
    """
    
    L_transformations = generate_list_transformations(transformations, thresh_indices, verbose=verbose)

    for transfo in L_transformations:
        patch = transfo(patch)
       
    return patch

def random_transformation(patch, thresh_indices = [0,0.5], transformations = {}, verbose=0):
    """
    :param patch: [image,mask].
    :param thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
    :return: application of a random transformation to the pair [image,mask].
    """
    
    L_transformations = generate_list_transformations(transformations, thresh_indices, verbose=verbose)
                    
    patch = random.choice(L_transformations)(patch)
       
    return patch

def labellize_mask_2d(patch, thresh_indices=[0, 0.5]):
    '''
    Process a patch with 8 bit pixels ([0-255]) so that the pixels between two threshold values are set to the closest threshold, effectively
    enabling the creation of a mask with as many different values as there are thresholds.

    Returns mask in [0-1] domain
    '''
    mask = np.zeros_like(patch)
    for indice in range(len(thresh_indices)-1):
        
        thresh_inf_8bit = 255*thresh_indices[indice]
        thresh_sup_8bit = 255*thresh_indices[indice+1]
        
        idx = np.where((patch >= thresh_inf_8bit) & (patch < thresh_sup_8bit)) # returns (x, y) of the corresponding indices
        mask[idx] = np.mean([thresh_inf_8bit/255,thresh_sup_8bit/255])
   
    mask[(patch >= 255*thresh_indices[-1])] = 1

    return patch


def transform_batches(list_batches):
    '''
    Transform batches so that they are readable by Tensorflow (good shapes)
    :param list_batches: [batch_x, batch_y, (batch_w)]
    :return transformed_batches: Returns the batches with good shapes for tensorflow
    '''
    batch_x = list_batches[0]
    batch_y = list_batches[1]
    if len(list_batches) == 3:
        batch_w = list_batches[2]
        
    if len(batch_y) == 1: # If we have only one image in the list np.stack won't work
        transformed_batches = []
        transformed_batches.append(np.reshape(batch_x[0], (1, batch_x[0].shape[0], batch_x[0].shape[1])))
        transformed_batches.append(np.reshape(batch_y[0], (1, batch_y[0].shape[0], batch_y[0].shape[1], -1)))
        
        if len(list_batches) == 3:
            transformed_batches.append(np.reshape(batch_w[0], (1, batch_w[0].shape[0], batch_w[0].shape[1])))
            
    else:
        transformed_batches = [np.stack(batch_x), np.stack(batch_y)]
         
        if len(list_batches) == 3:
            transformed_batches.append(np.stack(batch_w))
        
    return transformed_batches


#######################################################################################################################
#                                             Input data for the U-Net                                                #
#######################################################################################################################
class input_data:
    """
    Data to feed the learning/validating of the CNN
    """

    def __init__(self, trainingset_path, config, type_ ='train', batch_size = 8, preload_all=True):
        """
        Input: 
            trainingset_path : string : path to the trainingset folder containing 2 folders Validation and Train
                                    with images and ground truthes.
            type_ : string 'train' or 'validation' : for the network's training.
            thresh_indices : list of float in [0,1] : the thresholds for the ground truthes labels.
            preload_all : if put to True, will load every image into the memory.
        Output:
            None.
        """

        if type_ == 'train' : # Data for train
            self.path = trainingset_path+'/Train/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])
            self.each_sample_once = False

        if type_ == 'validation': # Data for validation
            self.path = trainingset_path+'/Validation/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])
            self.each_sample_once = True

        self.size_image = config["trainingset_patchsize"]
        self.n_labels = 2
        self.samples_seen = 0
        self.thresh_indices = config["thresholds"]
        self.batch_size = batch_size
        self.samples_list = self.reset_set(type_=type_)
        self.epoch_size = len(self.samples_list)
        self.preload_all = preload_all
        self.loaded_data = None
        #self.mean = config['dataset_mean']
        #self.variance = config['dataset_variance']

        # Loading all images if asked so
        
        if preload_all:
            self.loaded_data = {}
            for id_image in self.samples_list:
                # We are reading directly the images. Range of values : 0-255
                image = self.read_image('image', id_image)
                mask = self.read_image('mask', id_image)
                self.loaded_data.update({str(id_image):[image,mask]})        

    def get_size(self):
        return self.set_size
    
    def reset_set(self, type_= 'train', shuffle=True):
        """
        Reset the set.
        :param shuffle: If True, the set is shuffled, so that each batch won't systematically contain the same images.
        :return list: List of ids of training samples
        """
        
        self.sample_seen = 0
        
        if type_ == 'train':
            # Generation of a shuffled list of images      
            samples_list = list(range(self.set_size))
            if shuffle:
                np.random.shuffle(samples_list)

            # Adding X images so that all batches have the same size.
            rem = self.set_size % self.batch_size
            if rem != 0:
                samples_list += np.random.choice(samples_list, self.batch_size - rem, replace=False).tolist()
        else:
            samples_list = list(range(self.set_size))
            
        return samples_list


    def next_batch(self, augmented_data_ = {'type':'None'}, each_sample_once=False, data_aug_verbose=0):
        """
        :param augmented_data: if True, each patch of the batch is randomly transformed with the data augmentation process.
        :return: The pair [batch_x (data), batch_y (prediction)] to feed the network.
        """
                
        batch_x = []
        batch_y = []
        
        # Set the range of indices
        # Read the image and mask files.
        for i in range(self.batch_size) :

            # We load the image and discretize the masks
            image, real_mask = self.prepare_image_mask()

            # We apply data augmentation
            augmented_data = copy.deepcopy(augmented_data_)
            image, real_mask = self.apply_data_augmentation([image, real_mask], augmented_data, data_aug_verbose)

            # Normalisation of the image
            image = apply_legacy_preprocess(image)
            #image = apply_preprocess(image, self.mean, self.variance)
            # We save the obtained image and mask.
            batch_x.append(image)
            batch_y.append(real_mask)
            
            # If we are at the end of an epoch, we reset the list of samples, so that during next epoch all sets will be different.
            if self.sample_seen == self.epoch_size:
                if each_sample_once:
                    self.samples_list = self.reset_set(type_ = 'validation')
                    break
                else:
                    self.samples_list = self.reset_set(type_ = 'train')
        
        # Ensuring that we do have np.arrays of the good size for batch_x and batch_y before returning them 
        return transform_batches([batch_x, batch_y])


    def next_batch_WithWeights(self, augmented_data_ = {'type':'None'},
                               weights_modifier = {'balanced_activate':True, 'balanced_weights':[1.1, 1, 1.3],
                                                   'boundaries_activate':False},
                               each_sample_once=False, data_aug_verbose=0):

        """
        :param weights_modifier:
        :param augmented_data: if True, each patch of the batch is randomly transformed with the data augmentation process.
        :return: The triplet [batch_x (data), batch_y (prediction), weights (based on distance to edges)] to feed the network.
        """
        batch_x = []
        batch_y = []
        batch_w = []

        for i in range(self.batch_size) :

            # We prepare the image and the corresponding mask by discretizing the mask.
            image, real_mask = self.prepare_image_mask()

            # Generation of the weights map
            real_weights = self.generate_weights_map(weights_modifier, real_mask)

            # Application of data augmentation
            augmented_data = copy.deepcopy(augmented_data_)

            image, real_mask, real_weights = self.apply_data_augmentation([image, real_mask, real_weights],
                                                                          augmented_data, data_aug_verbose)
            
            # Normalisation of the image
            image = apply_legacy_preprocess(image)

            # We have now loaded the good image, a mask (under the shape of a matrix, with different labels) that still needs to be converted to a volume (meaning, a sparse cube where each layer of depth relates to a class)
            
            batch_x.append(image)
            batch_y.append(real_mask)
            batch_w.append(real_weights)
            
            # If we are at the end of an epoch, we reset the list of samples, so that during next epoch all sets will be different.
            if self.sample_seen == self.epoch_size:
                if each_sample_once:
                    self.samples_list = self.reset_set(type_ = 'validation')
                    break
                else:
                    self.samples_list = self.reset_set(type_ = 'train')
                    

        # Ensuring that we do have np.arrays of the good size for batch_x and batch_y before returning them        
        return transform_batches([batch_x, batch_y, batch_w])
    
    def read_image(self, type_, i):
        '''
        :param i: indice of the image or mask to read.
        :return image: the loaded image with 8 bit pixels, range of values being [0,288] 
        '''

        # Loading the image using 8-bit pixels (0-255)
        return imread(os.path.join(self.path,str(type_) + '_%s.png' % i), flatten=False, mode='L')

    def prepare_image_mask(self):
        """
        Loads the image and the mask, and discretizes the mask (and converts it in N dimensions, one for each class).
        :return: Image (ndarray, (H,W)) and Mask (ndarray, (H,W,C)). C number of classes.
        """

        # We take the next sample to see
        indice = self.samples_list.pop(0)
        self.sample_seen += 1

        if self.preload_all:
            image, mask = self.loaded_data[str(indice)]
        else:
            image = self.read_image('image', indice)
            mask = self.read_image('mask', indice)

        # Discretization of the mask
        mask = labellize_mask_2d(mask, self.thresh_indices) # mask intensity float between 0-1

        # Working out the real mask (sparse cube with n depth layer, one for each class)
        n = len(self.thresh_indices) # number of classes
        thresh_indices = [255*x for x in self.thresh_indices]
        real_mask = np.zeros([mask.shape[0], mask.shape[1], n])

        for class_ in range(n-1):
            real_mask[:,:,class_] = (mask[:,:] >= thresh_indices[class_]) * (mask[:,:] <  thresh_indices[class_+1])
        real_mask[:,:,-1] = (mask[:,:] >= thresh_indices[-1])
        real_mask = real_mask.astype(np.uint8)

        return [image, real_mask]


    def apply_data_augmentation(self, element, augmented_data, data_aug_verbose=0):
        """
        Applies data augmentation to the requested image and mask.
        :param image: Image (ndarray) to apply data augmentation to.
        :param mask: Mask of the image (ndarray) to apply data augmentation to.
        :param augmented_data: Dict, contains the parameters of the data augmentation to apply.
        :param data_aug_verbose: Int. If >=1, displays information about the data augmentation process.
        :return: Image and Mask that have been transformed.
        """


        # Online data augmentation
        if augmented_data['type'].lower() == 'all':
            augmented_data.pop('type')
            augmented_element = all_transformations(element,
                                                transformations=augmented_data,
                                                thresh_indices=self.thresh_indices,
                                                verbose=data_aug_verbose)

        elif augmented_data['type'].lower() == 'random':
            augmented_data.pop('type')
            augmented_element = random_transformation(element,
                                                  transformations=augmented_data,
                                                  thresh_indices=self.thresh_indices,
                                                  verbose=data_aug_verbose)

        else:
            augmented_element = element

        return augmented_element

    def generate_boundary_weights(self, real_mask, weights_intermediate, sigma):
        """
        Generates the boundary weights from the mask.
        :param real_mask: the discretized mask.
        :return: The 3D ndarray of the boundary weights (H,W,C) with C the number of classes.
        """

        # Create a weight map for each class (background is the first class, equal to 1
        n_classes = len(self.thresh_indices)

        # Classical method to compute weights
        for indice, class_ in enumerate(self.thresh_indices[1:]):

            mask_class = real_mask[:, :, indice]
            mask_class_8bit = np.asarray(255 * mask_class, dtype='uint8')
            weight = ndimage.distance_transform_edt(mask_class_8bit)
            weight[weight == 0] = np.max(weight)

            if class_ == self.thresh_indices[1]:
                w0 = 0.5
            else:
                w0 = 1

            weight = 1 + w0 * np.exp(-(weight.astype(np.float64) / sigma) ** 2 / 2)
            weights_intermediate[:, indice] = weight.reshape(-1, 1)[:, 0]

        # Generating the mask with the real labels as well as the matrix of the weights
        return np.reshape(weights_intermediate, [real_mask.shape[0], real_mask.shape[1], n_classes])


    def generate_weights_map(self, weights_modifier, real_mask):
        """
        Generates the weights for an image based on the mask.
        :param weights_modifier: Dict, contains the parameters about the weights to use.
        :param real_mask: Discretized mask (ndarray).
        :return: Weights map taking into account both balance weights and boundary weights.
        """
        weights_intermediate = np.ones((self.size_image * self.size_image, len(self.thresh_indices)))
        n = len(self.thresh_indices)

        # We generate the boundary weights map if necessary.
        if weights_modifier['boundaries_activate'] == True:

            # Create a boundary weight map for each class (background is the first class, equal to 1
            weights_intermediate = self.generate_boundary_weights(real_mask, weights_intermediate,
                                                                  weights_modifier['boundaries_sigma'])

        # Working out the real weights (sparse matrix with the weights associated with each pixel).
        # We apply the balance weights as well as the boundary weights if necessary.
        real_weights = np.zeros([real_mask.shape[0], real_mask.shape[1]])

        for class_ in range(n):
            mean_weights = np.mean(weights_modifier['balanced_weights'])
            weights_multiplier = 1
            if weights_modifier['balanced_activate'] == True:
                balanced_factor = weights_modifier['balanced_weights'][class_] / mean_weights
                weights_multiplier = np.multiply(weights_multiplier, balanced_factor)
            if weights_modifier['boundaries_activate'] == True:
                weights_multiplier = np.multiply(weights_multiplier, weights_intermediate[:, :, class_])

            real_weights += np.multiply(real_mask[:, :, class_], weights_multiplier)

        return real_weights


