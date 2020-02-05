


from skimage.transform import rescale
import numpy as np
from tqdm import tqdm
import shutil

import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.data_management.input_data import labellize_mask_2d
from AxonDeepSeg.data_management.patch_extraction import extract_patch
from AxonDeepSeg.ads_utils import convert_path

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

    # If string, convert to Path objects
    path_raw_data = convert_path(path_raw_data)
    path_patched_data = convert_path(path_patched_data)

    # First we define where we are going to store the patched data and we create the directory if it does not exist.
    if not path_patched_data.exists():
        path_patched_data.mkdir(parents=True)

    # Loop over each raw image folder
    img_folder_names = [im.name for im in path_raw_data.iterdir()]
    for img_folder in tqdm(img_folder_names):
        path_img_folder = path_raw_data / img_folder
        if path_img_folder.is_dir():

            # We are now in the image folder.
            file = open(path_img_folder / 'pixel_size_in_micrometer.txt', 'r')
            pixel_size = float(file.read())
            resample_coeff = float(pixel_size) / resampling_resolution # Used to set the resolution to the general_pixel_size

            # We go through every file in the image folder
            data_names = [d.name for d in path_img_folder.iterdir()]
            for data in data_names:
                if 'image' in data: # If it's the raw image.

                    img = ads.imread(path_img_folder / data)
                    img = rescale(img, resample_coeff, preserve_range=True, mode='constant').astype(int)

                elif 'mask' in data:
                    mask_init = ads.imread(path_img_folder / data)
                    mask = rescale(mask_init, resample_coeff, preserve_range=True, mode='constant', order=0)

                    # Set the mask values to the classes' values
                    mask = labellize_mask_2d(mask, thresh_indices)  # shape (size, size), values float 0.0-1.0

            to_extract = [img, mask]
            patches = extract_patch(to_extract, patch_size)
            # The patch extraction is done, now we put the new patches in the corresponding folders

            # We create it if it does not exist
            path_patched_folder = path_patched_data / img_folder
            if not path_patched_folder.exists():
                path_patched_folder.mkdir(parents=True)

            for j, patch in enumerate(patches):
                ads.imwrite(path_patched_folder.joinpath('image_%s.png'%j), patch[0])
                ads.imwrite(path_patched_folder.joinpath('mask_%s.png'%j), patch[1])

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

    # If string, convert to Path objects
    path_patched_data = convert_path(path_patched_data)
    path_dataset = convert_path(path_dataset)

    # Using the randomseed fed so that given a fixed input, the generation of the datasets is always the same.
    np.random.seed(random_seed)

    # First we define where we are going to store the patched data
    if not path_dataset.exists():
        path_dataset.mkdir(parents=True)

    # First case: there is only one type of acquisition to use.
    if type_ == 'unique':

        i = 0 # Total patches index

        # We loop through all folders containing patches
        patches_folder_names = [f for f in path_patched_data.iterdir()]
        for patches_folder in tqdm(patches_folder_names):

            path_patches_folder = path_patched_data / patches_folder.name
            if path_patches_folder.is_dir():

                # We are now in the patches folder
                L_img, L_mask = [], []
                filenames = [f for f in path_patches_folder.iterdir()]
                for data in filenames:
                    root, index = data.stem.split('_')

                    if 'image' in data.name:
                        img = ads.imread(path_patches_folder / data.name)
                        L_img.append((img, int(index)))

                    elif 'mask' in data.name:
                        mask = ads.imread(path_patches_folder / data.name)
                        L_mask.append((mask, int(index)))

                # Now we sort the patches to be sure we get them in the right order
                L_img_sorted, L_mask_sorted = sort_list_files(L_img, L_mask)

                # Saving the images in the new folder
                for img,k in L_img_sorted:
                    ads.imwrite(path_dataset.joinpath('image_%s.png'%i), img)
                    ads.imwrite(path_dataset.joinpath('mask_%s.png'%i), L_mask_sorted[k][0])
                    i = i+1 # Using the global i here.

    # Else we are using different types of acquisitions. It's important to have them separated in a SEM folder
    # and in a TEM folder.
    elif type_ == 'mixed':
        # We determine which acquisition type we are going to upsample (understand : take the same images multiple times)
        SEM_patches_folder = path_patched_data / 'SEM'
        TEM_patches_folder = path_patched_data / 'TEM'

        minority_patches_folder, len_minority, majority_patches_folder, len_majority = find_minority_type(
            SEM_patches_folder, TEM_patches_folder)

        # First we move all patches from the majority acquisition type to the new dataset
        foldernames = [folder.name for folder in majority_patches_folder.iterdir()]
        i = 0
        for patches_folder in tqdm(foldernames):

            path_patches_folder = majority_patches_folder / patches_folder
            if path_patches_folder.is_dir():
                # We are now in the patches folder
                L_img, L_mask = [], []
                filenames = [f for f in path_patches_folder.iterdir()]
                for data in path_patches_folder.iterdir():
                    root, index = data.stem.split('_')

                    if 'image' in data.name:
                        img = ads.imread(path_patches_folder / data.name)
                        L_img.append((img, int(index)))

                    elif 'mask' in data.name:
                        mask = ads.imread(path_patches_folder / data.name)
                        L_mask.append((mask, int(index)))
                # Now we sort the patches to be sure we get them in the right order
                L_img_sorted, L_mask_sorted = sort_list_files(L_img, L_mask)

                # Saving the images in the new folder
                for img,k in L_img_sorted:
                    ads.imwrite(path_dataset.joinpath('image_%s.png'%i), img)
                    ads.imwrite(path_dataset.joinpath('mask_%s.png'%i), L_mask_sorted[k][0])
                    i = i+1
        # Then we stratify - oversample the minority acquisition to the new dataset

        # We determine the ratio to take
        ratio_oversampling = float(len_majority)/len_minority

        # We go through each image folder in the minorty patches
        foldernames = [folder.name for folder in minority_patches_folder.iterdir()]
        for patches_folder in tqdm(foldernames):

            path_patches_folder = minority_patches_folder / patches_folder
            if path_patches_folder.is_dir():

                # We are now in the patches folder
                # We load every image
                filenames = [f for f in path_patches_folder.iterdir()]
                n_img = np.floor(len(filenames)/2)

                for data in filenames:
                    root, index = data.stem.split('_')
                    if 'image' in data.name:
                        img = ads.imread(path_patches_folder / data.name)
                        L_img.append((img, int(index)))

                    elif 'mask' in data.name:
                        mask = ads.imread(path_patches_folder / data.name)
                        L_mask.append((mask, int(index)))

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
                    ads.imwrite(path_dataset.joinpath('image_%s.png'%i), img)
                    ads.imwrite(path_dataset.joinpath('mask_%s.png'%i), mask)
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

    SEM_len = len([f for f in SEM_patches_folder.rglob("*") if f.is_file()])
    TEM_len = len([f for f in TEM_patches_folder.rglob("*") if f.is_file()])

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

def split_data(raw_dir, out_dir, seed=2019, split = [0.8, 0.2], override=False):
    """
    Splits a dataset into training and validation folders. 
    :param raw_dir: Raw dataset folder containing a set of subdirectories to be split.
    :param out_dir: Output directory of splitted dataset
    :param seed: Random number generator seed.
    :param split: Split fractions ([Train, Validation]).
    :param override: Bool. If out_dir exists and is True, the directory will be overwritten.
    """
    
    # If string, convert to Path objects
    raw_dir = convert_path(raw_dir)
    out_dir = convert_path(out_dir)
    
    # Handle case if output dir already exists
    if out_dir.is_dir():
        if override == True:
            shutil.rmtree(out_dir)
        else:
            raise IOError("Directory " + str(out_dir) + " already exist. Delete or change override option.")
    
    
    # Get splits for Training and Validation
    split_train = split[0]
    split_valid = split[1]

    # get sorted list of image directories
    dirs=sorted([x for x in raw_dir.iterdir() if x.is_dir()])

    # Create directories for splitted datasets
    train_dir = out_dir / "Train"
    train_dir.mkdir(parents=True, exist_ok=True)

    valid_dir = out_dir / "Validation"
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Randomly assign a number to each image folder for splitting
    np.random.seed(seed=seed)
    sorted_indices=np.random.choice(len(dirs), size=len(dirs), replace=False)

    # Move the image dirs from the raw folder to the split folder
    for index in sorted_indices[:round(len(sorted_indices)*split_train)]:
        dirs[index].rename(train_dir / dirs[index].parts[1])

    for index in sorted_indices[-round(len(sorted_indices)*split_valid):]:
        dirs[index].rename(valid_dir / dirs[index].parts[1])
