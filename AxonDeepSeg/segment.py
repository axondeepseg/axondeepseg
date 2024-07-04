
# Segmentation script
# -------------------
# This script lets the user segment automatically one or many images based on the segmentation models: SEM,
# TEM or BF.
#
# Maxime Wabartha - 2017-08-30

# Imports

from math import ceil
import os
import sys
from pathlib import Path

import json
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm
import pkg_resources
from typing import Literal, List, NoReturn
from loguru import logger

# AxonDeepSeg imports
import AxonDeepSeg
import AxonDeepSeg.ads_utils as ads
import AxonDeepSeg.zoom_factor_sweep as zfs
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.ads_utils import (convert_path, get_file_extension, 
                                   get_imshape, imwrite, imread)
import AxonDeepSeg.ads_utils
from config import axonmyelin_suffix, axon_suffix, myelin_suffix, valid_extensions

# Global variables
DEFAULT_MODEL_NAME = "model_seg_generalist_light"
MODELS_PATH = pkg_resources.resource_filename('AxonDeepSeg', 'models')
MODELS_PATH = Path(MODELS_PATH)

DEFAULT_MODEL_PATH = MODELS_PATH / DEFAULT_MODEL_NAME


def get_model_type(path_model: Path) -> Literal['light', 'ensemble']:
    '''
    Checks if the model has a "fold_all" model or is an ensemble of N-folds.
    '''
    if (path_model / 'fold_all').exists():
        return 'light'
    else:
        return 'ensemble'
    
def get_model_input_format(path_model: Path) -> tuple[str, int]:
    '''
    Get the input format of the model used for segmentation (e.g. '.png')

    Parameters
    ----------
    path_model : Path
        Path to the folder containing the model.
    
    Returns
    -------
    (format, n_channels): tuple[str, int]
        model input format (e.g. '.png') and nb. of input channels
    '''
    with open(path_model / 'dataset.json') as f:
        dataset_dict = json.load(f)
    fmt = dataset_dict['file_ending']
    channels = list(dataset_dict['channel_names'].keys())
    return fmt, len(channels)

@logger.catch
def prepare_inputs(path_imgs: List[Path], file_format: str, n_channels: int) -> List[Path]:
    '''
    Verifies if the input images can be sent to axon_segmentation(). Otherwise, 
    converts and saves in expected format.

    Parameters
    ----------
    path_imgs : List(Path)
        List of paths to the images to prepare.
    file_format : str
        Expected file format for the images, e.g. '.png' or '.tif'.
    n_channels : int
        Number of channels expected by the model, e.g. 1 for grayscale.

    Error
    -----
    If n_channels > 1 and the image doesn't have enough channels, an error is raised.
    Note that if n_channels == 1, the image will be converted to grayscale if needed.

    Returns
    -------
    filelist : List(Path)
        List of paths to sanitized images. If an image is already in the 
        expected format, the path is the same as the one in path_imgs. Otherwise, 
        will be the path to the converted image.
    '''
    filelist = []
    for im_path in path_imgs:
        target = im_path

        imshape = get_imshape(str(target)) # HWC format
        is_correct_shape = imshape[-1] == n_channels
        is_correct_format = target.suffix == file_format
        needs_conversion = not is_correct_shape or not is_correct_format
        
        if not is_correct_shape and n_channels != 1:
            logger.error(f'{str(target)} has {imshape[-1]} channels, expected {n_channels}.')
            sys.exit(2)

        if needs_conversion:
            im = imread(str(target))
            filename = target.stem
            if not is_correct_shape:
                logger.warning(f'{filename} will be converted to grayscale.')
                # add grayscale suffix to avoid overwriting original file
                target = Path(str(target.with_suffix('')) + '_grayscale' + file_format)
            if not is_correct_format:
                logger.warning(f'{filename} will be converted in the expected {file_format} format.')
                target = target.with_suffix(file_format)
            imwrite(str(target), im, file_format)
        
        filelist.append(target)

    return filelist 

@logger.catch
def segment_images(
        path_images: List[Path],
        path_model: Path,
        gpu_id: int=-1,
        verbosity_level: int=0,
    ) -> NoReturn:
    '''
    Segment the image(s) in path_images.

    Parameters
    ----------
    path_images : List(pathlib.Path)
        List of path(s) to the image(s) to segment.
    path_model : str or Path
        Path to the folder containing the model.
    gpu_id : int
        Number representing the GPU ID. Defaults to -1 for cpu.
    verbosity_level : int
        The higher, the more information is given about the segmentation process.
    '''

    path_images = [convert_path(p) for p in path_images]
    path_model = convert_path(path_model)
    (fileformat, n_channels) = get_model_input_format(path_model)
        
    for path_img in path_images:
        if not path_img.exists():
            logger.error(f"File {path_img} does not exist.")
            sys.exit(2)
    path_images_sanitized = prepare_inputs(path_images, fileformat, n_channels)
    
    axon_segmentation(
        path_inputs=path_images_sanitized, 
        path_model=path_model, 
        model_type=get_model_type(path_model),
        gpu_id=gpu_id, 
        verbosity_level=verbosity_level
    )

@logger.catch
def segment_folders(path_testing_images_folder, 
                    path_model,
                    gpu_id=0,
                    verbosity_level=0):
    '''
    Segments the images contained in the image folders located in the path_testing_images_folder.
    :param path_testing_images_folder: the folder where all image folders are located (the images to segment are located
    in those image folders)
    :param path_model: where to access the model.
    :param overlap_value: the number of pixels to be used for overlap when doing prediction. Higher value means less
    border effects but more time to perform the segmentation.
    :param acquired_resolution: isotropic pixel resolution of the acquired images.
    :param zoom_factor: multiplicative constant applied to the pixel size before model inference.
    :param no_patch: If True, the image is segmented without using patches. Default: False.
    :param gpu_id: Number representing the GPU ID for segmentation if available. Default: 0.
    :param verbosity_level: Level of verbosity. The higher, the more information is given about the segmentation
    process.
    :return: Nothing.
    '''
    logger.info(f'Starting segmentation of multiple images in "{Path(path_testing_images_folder).resolve()}".')

    # If string, convert to Path objects
    path_testing_images_folder = convert_path(path_testing_images_folder)
    path_model = convert_path(path_model)

    # Update list of images to segment by selecting only image files (not already segmented or not masks)
    img_files = [file for file in path_testing_images_folder.iterdir() if (file.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'))
                 and (not str(file).endswith((str(axonmyelin_suffix), str(axon_suffix), str(myelin_suffix),'mask.png')))]

     # Get the model's native resolution
    model_resolution, patch_size = get_model_native_resolution_and_patch(path_model)

    # Pre-processing: convert to png if not already done and adapt to model contrast
    for file_ in tqdm(img_files, desc="Segmentation..."):

        # Get pixel size from file, if needed
        # If we did not receive any resolution we read the pixel size in micrometer from each pixel.
        if acquired_resolution == None:
            if (path_testing_images_folder / 'pixel_size_in_micrometer.txt').exists():
                resolutions_file = open(path_testing_images_folder / 'pixel_size_in_micrometer.txt', 'r')
                str_resolution = float(resolutions_file.read())
                acquired_resolution = float(str_resolution)
            else:
                exception_msg = "ERROR: No pixel size is provided, and there is no pixel_size_in_micrometer.txt file in image folder. " \
                                "Please provide a pixel size (using argument acquired_resolution), or add a pixel_size_in_micrometer.txt file " \
                                "containing the pixel size value."
                logger.error(exception_msg)
                raise Exception(exception_msg)

        logger.info(f"Loading {path_testing_images_folder / file_}.")
        try:
            height, width, _ = ads.imread(str(path_testing_images_folder / file_)).shape
        except:
            try:
                height, width = ads.imread(str(path_testing_images_folder / file_)).shape
            except Exception as e:
                raise e

        image_size = [height, width]

        selected_model = path_model.name

        # Read image for conversion
        img = ads.imread(str(path_testing_images_folder / file_))

        img_name_original = file_.stem

        acquisition_name = file_.name
       

        # Check that the resampled image will be of sufficient size, and if not throw an error.
        h, w = image_size

        w_resampled = w*(acquired_resolution*zoom_factor)/model_resolution
        h_resampled = h*(acquired_resolution*zoom_factor)/model_resolution

        if w_resampled < patch_size or h_resampled < patch_size:
            if w<=h:
                minimum_zoom_factor = patch_size*model_resolution/(w*acquired_resolution)
            else:
                minimum_zoom_factor = patch_size*model_resolution/(h*acquired_resolution)

            # Round to 1 decimal, always up.
            minimum_zoom_factor = ceil(minimum_zoom_factor*10)/10

            warning_msg = "Skipping image: Due to your given image size, resolution, and zoom factor, the image " \
                   f"{path_testing_images_folder / file_}" \
                   " is smaller than the patch size after it is resampled during segmentation. " \
                   "To resolve this, please set a zoom factor greater than " \
                   f"{minimum_zoom_factor}  for this image on a re-run. " \
                   "To do this on the command line, call the segmentation with the -z flag, i.e. " \
                   f"-z {minimum_zoom_factor}"
            logger.info(warning_msg)
        else:
            axon_segmentation(path_acquisitions_folders=path_testing_images_folder,
                            acquisitions_filenames=[str(path_testing_images_folder  / acquisition_name)],
                            path_model_folder=path_model, acquired_resolution=acquired_resolution*zoom_factor,
                            overlap_value=overlap_value, no_patch=no_patch, gpu_id=gpu_id)
            if verbosity_level >= 1:
                tqdm.write("Image {0} segmented.".format(str(path_testing_images_folder / file_)))


    logger.info("Segmentations done.")
    return None

# Main loop
def main(argv=None):
    '''
    Main loop.
    :return: Exit code.
        0: Success
        1: Invalid extension
        2: Invalid input
    '''
    logger.add("axondeepseg.log", level='DEBUG', enqueue=True)
    logger.info(f"AxonDeepSeg v.{AxonDeepSeg.__version__}")

    ap = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    requiredName = ap.add_argument_group('required arguments')

    # Setting the arguments of the segmentation
    requiredName.add_argument(
        '-i', '--imgpath', 
        required=True, 
        nargs='+', 
        help='Path to the image to segment or path to the folder \n'
            + 'where the image(s) to segment is/are located.',
    )
    ap.add_argument(
        "-m", "--model", 
        required=False, 
        help='Folder where the model is located, if different from the default model.',
    )
    ap.add_argument(
        '-v', '--verbose', 
        required=False, 
        type=int, 
        choices=list(range(0,2)), 
        help='Verbosity level. \n'
            + '0 (default) : Quiet mode. Shows minimal information on the terminal. \n'
            + '1: Developer mode. Shows more information on the terminal, useful for debugging.',
        default=0,
    )
    ap.add_argument(
        "--gpu-id",
        dest="gpu_id",
        required=False,
        type=int,
        help='Number representing the GPU ID for segmentation if available. Default: -1 (cpu).',
        default=-1,
    )
    ap._action_groups.reverse()

    # Processing the arguments
    args = vars(ap.parse_args(argv))
    verbosity_level = int(args["verbose"])
    path_target_list = [Path(p) for p in args["imgpath"]]
    path_model = Path(args["model"]) if args["model"] else DEFAULT_MODEL_PATH
    gpu_id = int(args["gpu_id"])

    # Check for available GPU IDs
    if gpu_id >= 0:
        ads.check_available_gpus(gpu_id)

    input_img_list = []
    input_dir_list = []
    # Separate image vs directory paths
    for current_path_target in path_target_list:
        if not current_path_target.is_dir():
            if not get_file_extension(current_path_target) in valid_extensions:
                logger.error(f"Invalid extension for file {current_path_target}.")
                sys.exit(1)
            input_img_list.append(current_path_target)
        else:
            input_dir_list.append(current_path_target)
    
    # perform segmentation
    if input_img_list:
        segment_images(input_img_list, path_model, gpu_id, verbosity_level)
    if input_dir_list:
        for dir_path in input_dir_list:
            segment_folders(dir_path, path_model, gpu_id, verbosity_level)

    sys.exit(0)

# Calling the script
if __name__ == '__main__':
    with logger.catch():
        main()
