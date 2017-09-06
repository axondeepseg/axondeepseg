import os
import shutil
from scipy.misc import imread, imsave
from skimage.transform import rescale
from AxonDeepSeg.data_management.input_data import labellize_mask_2d
import random
import numpy as np
from patch_extraction import extract_patch
from tqdm import tqdm


def raw_img_to_patches(path_raw_data, path_patched_data, thresh_indices = [0, 0.2, 0.8], random_seed = None, patch_size=256, general_pixel_size=0.2):
    '''
    Note: this functions needs to be run as many times as there are different general pixel size (thus different acquisition types / resolutions).
    '''
    # First we define where we are going to store the patched data
    if not os.path.exists(path_patched_data):
        os.makedirs(path_patched_data)
        
    # Loop over each raw image folder
    for img_folder in tqdm(os.listdir(path_raw_data)):
        path_img_folder = os.path.join(path_raw_data, img_folder)
        if os.path.isdir(path_img_folder):
            # We are now in the image folder.
            file = open(path_img_folder+'/pixel_size_in_micrometer.txt', 'r')
            pixel_size = float(file.read())
            rescale_coeff = pixel_size/general_pixel_size # Used to set the resolution to the general_pixel_size
            
            # We go through every file in the image folder
            for data in os.listdir(path_img_folder):
                if 'image' in data: # If it's the raw image.
                    img = imread(os.path.join(path_img_folder, data), flatten=False, mode='L')
                    img = rescale(img, rescale_coeff, preserve_range=True).astype(int)
                  
                elif 'mask.png' in data:
                    mask_init = imread(os.path.join(path_img_folder, data), flatten=False, mode='L')
                    mask = rescale(mask_init, rescale_coeff, preserve_range=True)

                    # Set the mask values to the classes' values
                    mask = labellize_mask_2d(mask, thresh_indices)  # shape (256, 256), values float 0.0-1.0

            patches = extract_patch(img, mask, patch_size)
            # The patch extraction is done, now we put the new patches in the corresponding folders
            
            # We create it if it does not exist
            path_patched_folder = os.path.join(path_patched_data,img_folder)
            if not os.path.exists(path_patched_folder):
                os.makedirs(path_patched_folder)
            
            for j, patch in enumerate(patches):
                imsave(os.path.join(path_patched_folder,'image_%s.png'%j), patch[0],'png')
                imsave(os.path.join(path_patched_folder,'mask_%s.png'%j), patch[1],'png')

def patched_to_dataset(path_patched_data, path_dataset, type_, random_seed = None):
    '''
    Note: Here a dataset is either a trainingset, a validation set or a testing set. For flexiblity purposes you will need to 
    '''
    
    np.random.seed(random_seed)
    # First we define where we are going to store the patched data
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)  
        
    if type_ == 'unique':
        i = 0 # Total patches index
        # We loop through all folders containing patches
        for patches_folder in tqdm(os.listdir(path_patched_data)):
            path_patches_folder = os.path.join(path_patched_data, patches_folder)
            if os.path.isdir(path_patches_folder):
                # We are now in the patches folder
                L_img, L_mask = [], []
                for data in os.listdir(path_patches_folder):
                    data_name = data[:-4].split('_')
                    if 'image' in data:
                        img = imread(os.path.join(path_patches_folder, data), flatten=True, mode='L')
                        L_img.append((img, int(data_name[-1])))

                    elif 'mask' in data:
                        mask = imread(os.path.join(path_patches_folder, data), flatten=True, mode='L')
                        L_mask.append((mask, int(data_name[-1])))
                # Now we sort the patches to be sure we get them in the right order
                L_img_sorted, L_mask_sorted = sort_list_files(L_img, L_mask)

                # Saving the images in the new folder
                for img,k in L_img_sorted:
                    imsave(os.path.join(path_dataset,'image_%s.png'%i), img, 'png')
                    imsave(os.path.join(path_dataset,'mask_%s.png'%i), L_mask_sorted[k][0], 'png')
                    i = i+1
                    
    elif type_ == 'mixed':
        i = 0
        # We determine which acquisition type we are going to upsample (understand : take the same images multiple times)
        SEM_patches_folder = os.path.join(path_patched_data,'SEM')
        TEM_patches_folder = os.path.join(path_patched_data,'TEM')
        
        SEM_len = sum([len(files) for r, d, files in os.walk(SEM_patches_folder)])
        TEM_len = sum([len(files) for r, d, files in os.walk(TEM_patches_folder)])
 
        if SEM_len < TEM_len:
            minority_patches_folder = SEM_patches_folder
            majority_patches_folder = TEM_patches_folder
            len_minority = SEM_len
            len_majority = TEM_len
        else:
            minority_patches_folder = TEM_patches_folder
            majority_patches_folder = SEM_patches_folder
            len_minority = TEM_len
            len_majority = SEM_len
                    
        # First we move all patches from the majority acquisition type to the new dataset
        for patches_folder in tqdm(os.listdir(majority_patches_folder)):
            path_patches_folder = os.path.join(majority_patches_folder, patches_folder)
            if os.path.isdir(path_patches_folder):
                # We are now in the patches folder
                L_img, L_mask = [], []
                for data in os.listdir(path_patches_folder):
                    data_name = data[:-4].split('_')
                    if 'image' in data:
                        img = imread(os.path.join(path_patches_folder, data), flatten=True, mode='L')
                        L_img.append((img, int(data_name[-1])))

                    elif 'mask' in data:
                        mask = imread(os.path.join(path_patches_folder, data), flatten=True, mode='L')
                        L_mask.append((mask, int(data_name[-1])))
                # Now we sort the patches to be sure we get them in the right order
                L_img_sorted, L_mask_sorted = sort_list_files(L_img, L_mask)

                # Saving the images in the new folder
                for img,k in L_img_sorted:
                    imsave(os.path.join(path_dataset,'image_%s.png'%i), img, 'png')
                    imsave(os.path.join(path_dataset,'mask_%s.png'%i), L_mask_sorted[k][0], 'png')
                    i = i+1
        # Then we stratify - oversample the minority acquisition to the new dataset
        
        # We determine the ratio to take
        ratio = float(len_majority)/len_minority

        # We go through each image folder in the minorty patches
        for patches_folder in tqdm(os.listdir(minority_patches_folder)):
            path_patches_folder = os.path.join(minority_patches_folder, patches_folder)
            if os.path.isdir(path_patches_folder):
                # We are now in the patches folder
                n_img = np.floor(len(os.listdir(path_patches_folder)[:])/2)
                
                # We load every image
                for data in os.listdir(path_patches_folder):
                    data_name = data[:-4].split('_')
                    if 'image' in data:
                        img = imread(os.path.join(path_patches_folder, data), flatten=True, mode='L')
                        L_img.append((img, int(data_name[-1])))

                    elif 'mask' in data:
                        mask = imread(os.path.join(path_patches_folder, data), flatten=True, mode='L')
                        L_mask.append((mask, int(data_name[-1])))
                        
                # Now we sort the patches to be sure we get them in the right order
                L_img_sorted, L_mask_sorted = sort_list_files(L_img, L_mask)
                L_merged_sorted = np.asarray([L_img_sorted[j] + L_mask_sorted[j] for j in range(len(L_img_sorted))])
                
                # We create a new array composed of enough elements so that the two types of acquisitions are balanced
                L_elements_to_save = L_merged_sorted[np.random.choice(int(L_merged_sorted.shape[0]),int(np.ceil(ratio*n_img)), replace=True),:]
                for j in range(L_elements_to_save.shape[0]):
                    img = L_elements_to_save[j][0]
                    mask = L_elements_to_save[j][2]
                    imsave(os.path.join(path_dataset,'image_%s.png'%i), img, 'png')
                    imsave(os.path.join(path_dataset,'mask_%s.png'%i), mask, 'png')
                    i = i+1            


def sort_list_files(list_patches, list_masks):
    # We sort the transformations to make by the number preceding the transformation in the dict in the config file        
    return sorted(list_patches, key=lambda x: int(x[1])), sorted(list_masks, key=lambda x: int(x[1]))
    
    









