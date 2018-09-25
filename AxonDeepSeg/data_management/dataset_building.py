import os
from scipy.misc import imread, imsave
from skimage.transform import rescale
from AxonDeepSeg.data_management.input_data import labellize_mask_2d
import numpy as np
from .patch_extraction import extract_patch
from tqdm import tqdm
import AxonDeepSeg.ads_utils


def raw_img_to_patches(path_raw_data, path_patched_data, thresh_indices = [0, 0.2, 0.8],
                       patch_size=512, resampling_resolution=0.1):
    """
    Transform a raw acquisition to a folder of patches of size indicated in the arguments. Also performs resampling.
    Note: this functions needs to be run as many times as there are different general pixel size
    (thus different acquisition types / resolutions).
    :param path_raw_data: Path to where the raw image folders are located.
    :param path_patched_data: Path to where we will store the patched acquisitions.
    :param thresh_indices: List of float, determining the thresholds separating the classes.
    :param patch_size: Int, size of the patches to generate (and consequently input size of the network).
    :param resampling_resolution: Float, the resolution we need to resample to so that each sample
    has the same resolution in a dataset.
    :return: Nothing.
    """

    # First we define where we are going to store the patched data and we create the directory if it does not exist.
    if not os.path.exists(path_patched_data):
        os.makedirs(path_patched_data)
        
    # Loop over each raw image folder
    for img_folder in tqdm(os.listdir(path_raw_data)):
        path_img_folder = os.path.join(path_raw_data, img_folder)
        if os.path.isdir(path_img_folder):

            # We are now in the image folder.
            file = open(path_img_folder+'/pixel_size_in_micrometer.txt', 'r')
            pixel_size = float(file.read())
            resample_coeff = float(pixel_size) / resampling_resolution # Used to set the resolution to the general_pixel_size
            
            # We go through every file in the image folder
            for data in os.listdir(path_img_folder):
                if 'image' in data: # If it's the raw image.
                    img = imread(os.path.join(path_img_folder, data), flatten=False, mode='L')
                    img = rescale(img, resample_coeff, preserve_range=True).astype(int)
                  
                elif 'mask.png' in data:
                    mask_init = imread(os.path.join(path_img_folder, data), flatten=False, mode='L')
                    mask = rescale(mask_init, resample_coeff, preserve_range=True)

                    # Set the mask values to the classes' values
                    mask = labellize_mask_2d(mask, thresh_indices)  # shape (size, size), values float 0.0-1.0

            to_extract = [img, mask]
            patches = extract_patch(to_extract, patch_size)
            # The patch extraction is done, now we put the new patches in the corresponding folders
            
            # We create it if it does not exist
            path_patched_folder = os.path.join(path_patched_data,img_folder)
            if not os.path.exists(path_patched_folder):
                os.makedirs(path_patched_folder)
            
            for j, patch in enumerate(patches):
                imsave(os.path.join(path_patched_folder,'image_%s.png'%j), patch[0],'png')
                imsave(os.path.join(path_patched_folder,'mask_%s.png'%j), patch[1],'png')

def patched_to_dataset(path_patched_data, path_dataset, type_, random_seed=None):
    """
    Creates a dataset using already created patches.
    :param path_patched_data: Path to where to find the folders where the patches folders are located.
    :param path_dataset: Path to where to create the newly formed dataset.
    :param type_: String, either 'unique' or 'mixed'. Unique means that we create a dataset with only TEM or only SEM
    data. "Mixed" means that we are creating a dataset with both type of images.
    :param random_seed: Int, the random seed to use to be able to consistenly recreate generated datasets.
    :return: None.
    """

    # Using the randomseed fed so that given a fixed input, the generation of the datasets is always the same.
    np.random.seed(random_seed)

    # First we define where we are going to store the patched data
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)  

    # First case: there is only one type of acquisition to use.
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
                    i = i+1 # Using the global i here.

    # Else we are using different types of acquisitions. It's important to have them separated in a SEM folder
    # and in a TEM folder.
    elif type_ == 'mixed':

        i = 0

        # We determine which acquisition type we are going to upsample (understand : take the same images multiple times)
        SEM_patches_folder = os.path.join(path_patched_data,'SEM')
        TEM_patches_folder = os.path.join(path_patched_data,'TEM')

        minority_patches_folder, len_minority, majority_patches_folder, len_majority = find_minority_type(
            SEM_patches_folder, TEM_patches_folder)

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
        ratio_oversampling = float(len_majority)/len_minority

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
                # (oversampling)
                L_elements_to_save = L_merged_sorted[np.random.choice(
                    int(L_merged_sorted.shape[0]),int(np.ceil(ratio_oversampling*n_img)), replace=True),:]

                # Finally we save all the images in order at the new dataset path.
                for j in range(L_elements_to_save.shape[0]):
                    img = L_elements_to_save[j][0]
                    mask = L_elements_to_save[j][2]
                    imsave(os.path.join(path_dataset,'image_%s.png'%i), img, 'png')
                    imsave(os.path.join(path_dataset,'mask_%s.png'%i), mask, 'png')
                    i = i+1            


def sort_list_files(list_patches, list_masks):
    """
    Sorts a list of patches and masks depending on their id.
    :param list_patches: List of name of patches in the folder, that we want to sort.
    :param list_masks: List of name of masks in the folder, that we want to sort.
    :return: List of sorted lists, respectively of patches and masks.
    """

    return sorted(list_patches, key=lambda x: int(x[1])), sorted(list_masks, key=lambda x: int(x[1]))
    

def find_minority_type(SEM_patches_folder, TEM_patches_folder):
    """
    Identifies the type of acquisition that has the least number of patches in order to manage oversampling after.
    :param SEM_patches_folder: Path to the SEM patches.
    :param TEM_patches_folder: Path to the TEM patches.
    :return: The path to the minority patches folder, the number of patches of the minority patches folder,
    the path to the majority path folder and the number of patches in this folder.
    """

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

    return minority_patches_folder, len_minority, majority_patches_folder, len_majority
