
# Segmentation script
# -------------------
# This script lets the user segment automatically one or many images based on the segmentation models: SEM,
# TEM or BF.
#
# Maxime Wabartha - 2017-08-30

# Imports

import sys
from pathlib import Path

import json
import argparse
from argparse import RawTextHelpFormatter
from typing import Literal, List, NoReturn
from loguru import logger

# AxonDeepSeg imports
import AxonDeepSeg
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.ads_utils import (convert_path, get_file_extension, 
                                   get_imshape, imwrite, imread)
import AxonDeepSeg.ads_utils
from AxonDeepSeg.params import valid_extensions, generated_file_suffixes

# Global variables
DEFAULT_MODEL_NAME = "model_seg_generalist_light"

package_dir = Path(AxonDeepSeg.__file__).parent
MODELS_PATH = package_dir / 'models'

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
def prepare_inputs(path_imgs: List[Path], file_format: str, n_channels: int, allow_large_images: bool = False) -> List[Path]:
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

        imshape = get_imshape(str(target), allow_large_images=allow_large_images)
        is_correct_shape = (imshape[-1] == n_channels)
        is_correct_format = (target.suffix == file_format)
        needs_conversion = (is_correct_shape == False) or (is_correct_format == False)

        if needs_conversion:
            if n_channels != 1:
                logger.error(f'{str(target)} has {imshape[-1]} channels, expected {n_channels}.')
                sys.exit(2)
            im = imread(str(target), allow_large_images=allow_large_images)
            filename = target.stem
            if is_correct_shape == False:
                logger.warning(f'{filename} will be converted to grayscale.')
                # add grayscale suffix to avoid overwriting original file
                target = Path(str(target.with_suffix('')) + '_grayscale' + file_format)
            if is_correct_format == False:
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
        allow_large_images: bool=False,
        backend: str='auto',
        tile_step_size: float=0.5,
    ) -> NoReturn:
    '''
    Segment the image(s) in path_images.

    Parameters
    ----------
    path_images : List(pathlib.Path)
        List of path(s) to the image(s) to segment.
    path_model : pathlib.Path
        Path to the folder containing the model.
    gpu_id : int
        Number representing the GPU ID. Defaults to -1 for cpu.
    verbosity_level : int
        The higher, the more information is given about the segmentation process.
    '''

    path_images = [convert_path(p) for p in path_images]
    path_model = convert_path(path_model)

    available_models = ads.get_existing_models_list()
    if not available_models or path_model.stem not in available_models:
            try:
                print('Model not found, attempting to download')
                # Call download models from the AxonDeepSeg/download_model.py module

                import AxonDeepSeg.download_model as download_model
                download_model.download_model()

                available_models = ads.get_existing_models_list()
                assert path_model.stem in available_models
            except Exception as e:
                raise Exception('Could not download models, try again.') from e

    (fileformat, n_channels) = get_model_input_format(path_model)
        
    for path_img in path_images:
        if not path_img.exists():
            logger.error(f"File {path_img} does not exist.")
            sys.exit(2)
    path_images_sanitized = prepare_inputs(path_images, fileformat, n_channels, allow_large_images=allow_large_images)
    if path_images_sanitized is None:
        return

    axon_segmentation(
        path_inputs=path_images_sanitized,
        path_model=path_model,
        model_type=get_model_type(path_model),
        gpu_id=gpu_id,
        verbosity_level=verbosity_level,
        allow_large_images=allow_large_images,
        backend=backend,
        tile_step_size=tile_step_size,
    )

def list_images_in_folder(path_folder: Path) -> List[Path]:
    '''
    List the images to segment in a folder, excluding masks and other files
    AxonDeepSeg has generated.

    Parameters
    ----------
    path_folder : pathlib.Path
        Path to the folder containing the images to segment.

    Returns
    -------
    List[pathlib.Path]
        Paths of the image files to segment.
    '''
    path_folder = convert_path(path_folder)
    return [
        file for file in path_folder.iterdir()
            if (file.suffix.lower() in valid_extensions)
            and not str(file).endswith(generated_file_suffixes)
    ]

@logger.catch
def segment_folder(
        path_folder: Path,
        path_model: Path,
        gpu_id: int=-1,
        verbosity_level: int=0,
        allow_large_images: bool=False,
        backend: str='auto',
        tile_step_size: float=0.5,
    ) -> NoReturn:
    '''
    Segments all images in the path_folder directory.

    Parameters
    ----------
    path_folder : pathlib.Path
        Path to the folder containing the images to segment
    path_model : pathlib.Path
        Path to the folder containing the model.
    gpu_id : int
        Number representing the GPU ID. Defaults to -1 for cpu.
    verbosity_level : int
        The higher, the more information is given about the segmentation process.
    '''
    logger.info(f'Starting segmentation of multiple images in "{str(path_folder)}".')

    # If string, convert to Path objects
    path_folder = convert_path(path_folder)
    path_model = convert_path(path_model)

    segment_images(list_images_in_folder(path_folder), path_model, gpu_id, verbosity_level,
                   allow_large_images=allow_large_images, backend=backend,
                   tile_step_size=tile_step_size)
    logger.info("Folder segmentation done.")

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
        help='Number representing the GPU ID for segmentation if available. Default: None (cpu).',
        default=-1,
    )
    ap.add_argument(
        "--allow-large-images",
        dest="allow_large_images",
        required=False,
        action='store_true',
        default=False,
        help='Allow segmentation of images exceeding PIL\'s default decompression bomb limit '
             f'(~178 Mpx). \nUse this for large microscopy acquisitions.',
    )
    ap.add_argument(
        "--backend",
        required=False,
        choices=['auto', 'torch', 'torch-fp16', 'coreml'],
        default='auto',
        help='Execution backend for the forward pass. \n'
             'auto (default): fastest available for this machine. \n'
             'coreml: Apple CoreML, uses the Neural Engine (macOS only, ~3.6x faster). \n'
             'torch-fp16: half precision through PyTorch (~1.7x faster on MPS). \n'
             'torch: full precision PyTorch, the historical behaviour. \n'
             'All backends produce near-identical output.',
    )
    ap.add_argument(
        "--tile-step-size",
        dest="tile_step_size",
        required=False,
        type=float,
        default=0.5,
        help='Sliding-window overlap between 0 and 1. Default 0.5 (each pixel covered \n'
             '~4x). 1.0 means no overlap: faster, slightly coarser.',
    )
    ap._action_groups.reverse()

    # Processing the arguments
    args = vars(ap.parse_args(argv))
    
    # Load log file without logger to write
    with open("axondeepseg.log", "a") as f:
        f.write("===================================================================================\n")

    logger.info(AxonDeepSeg.__version_string__)
    logger.info(f"Command line arguments: {args}")

    verbosity_level = int(args["verbose"])
    path_target_list = [Path(p) for p in args["imgpath"]]
    path_model = Path(args["model"]) if args["model"] else DEFAULT_MODEL_PATH
    allow_large_images = args["allow_large_images"]

    gpu_id = int(args["gpu_id"])
    backend = args["backend"]
    tile_step_size = float(args["tile_step_size"])
    if not 0 < tile_step_size <= 1:
        logger.error("--tile-step-size must be in (0, 1].")
        sys.exit(2)

    # Check for available GPU IDs
    if gpu_id >=0:
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
    
    # Segment everything in a single batch so the model is loaded once, rather
    # than once per folder.
    for dir_path in input_dir_list:
        logger.info(f'Collecting images in "{str(dir_path)}".')
        input_img_list.extend(list_images_in_folder(dir_path))

    if input_img_list:
        segment_images(input_img_list, path_model, gpu_id, verbosity_level,
                       allow_large_images=allow_large_images, backend=backend,
                       tile_step_size=tile_step_size)
    else:
        logger.warning('No images to segment.')

    sys.exit(0)

# Calling the script
if __name__ == '__main__':
    with logger.catch():
        main()
