
# Segmentation script
# -------------------
# This script lets the user segment automatically one or many images based on the segmentation models: SEM,
# TEM or BF.
#
# Maxime Wabartha - 2017-08-30

# Imports

from math import ceil
import os
from os import error
import sys
from pathlib import Path

import json
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm
import pkg_resources
from PIL import Image

# AxonDeepSeg imports
import AxonDeepSeg
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.ads_utils import convert_path
from config import axonmyelin_suffix, axon_suffix, myelin_suffix

# Global variables
SEM_DEFAULT_MODEL_NAME = "model_seg_rat_axon-myelin_sem"
TEM_DEFAULT_MODEL_NAME = "model_seg_mouse_axon-myelin_tem"
BF_DEFAULT_MODEL_NAME = "model_seg_rat_axon-myelin_bf"

MODELS_PATH = pkg_resources.resource_filename('AxonDeepSeg', 'models')
MODELS_PATH = Path(MODELS_PATH)

default_SEM_path = MODELS_PATH / SEM_DEFAULT_MODEL_NAME
default_TEM_path = MODELS_PATH / TEM_DEFAULT_MODEL_NAME
default_BF_path = MODELS_PATH / BF_DEFAULT_MODEL_NAME

default_overlap = 48

# Definition of the functions

def generate_default_parameters(type_acquisition, new_path):
    '''
    Generates the parameters used for segmentation for the default model corresponding to the type_model acquisition.
    :param type_model: String, the type of model to get the parameters from.
    :param new_path: Path to the model to use.
    :return: the config dictionary.
    '''
    
    # If string, convert to Path objects
    new_path = convert_path(new_path)

    # Building the path of the requested model if it exists and was supplied, else we load the default model.
    if type_acquisition == 'SEM':
        if (new_path is not None) and new_path.exists():
            path_model = new_path
        else:
            path_model = MODELS_PATH / SEM_DEFAULT_MODEL_NAME
    elif type_acquisition == 'TEM':
        if (new_path is not None) and new_path.exists():
            path_model = new_path
        else:
            path_model = MODELS_PATH / TEM_DEFAULT_MODEL_NAME
    elif type_acquisition == 'BF':
        if (new_path is not None) and new_path.exists():
            path_model = new_path
        else:
            path_model = MODELS_PATH / BF_DEFAULT_MODEL_NAME
    else:
        raise ValueError

    return path_model

def get_model_native_resolution_and_patch(path_model):
    '''
    Get the native resolution of the model, ie. the resolution that segmented images will get resampled to.
    Also, get the patch size.
    :param path_model: model directory
    :return: model_resolution, patch_size
    '''
    resolution_unit_conversion_factor = 1e3 # IVADOMED uses a native mm resolution, whereas ADS uses um
    
    model_json = [pos_json for pos_json in os.listdir(path_model) if pos_json.endswith('.json')]
    model = json.load(open(path_model / model_json[0]))

    model_resolution = [model['transformation']['Resample']['wspace'], model['transformation']['Resample']['hspace']]
    model_resolution[0] = model_resolution [0] * resolution_unit_conversion_factor
    model_resolution[1] = model_resolution [1] * resolution_unit_conversion_factor
    
    patch_size = [model['default_model']['length_2D'][0], model['default_model']['length_2D'][1]]

    if model_resolution[0] == model_resolution[1]: #isotropic pixels
        model_resolution = model_resolution[0]

    if patch_size[0] == patch_size[1]: #isotropic pixels
        patch_size = patch_size[0]

    return model_resolution, patch_size

def segment_image(
                path_testing_image,
                path_model,
                overlap_value,
                acquired_resolution = None,
                zoom_factor = 1.0,
                verbosity_level = 0):

    '''
    Segment the image located at the path_testing_image location.
    :param path_testing_image: the path of the image to segment.
    :param path_model: where to access the model
    :param overlap_value: the number of pixels to be used for overlap when doing prediction. Higher value means less
    border effects but more time to perform the segmentation.
    :param acquired_resolution: isotropic pixel resolution of the acquired images.
    :param zoom_factor: multiplicative constant applied to the pixel size before model inference.
    :param verbosity_level: Level of verbosity. The higher, the more information is given about the segmentation
    process.
    :return: Nothing.
    '''

    # If string, convert to Path objects
    path_testing_image = convert_path(path_testing_image)
    path_model = convert_path(path_model)

    if path_testing_image.exists():

        # Extracting the image name and its folder path from the total path.
        path_parts = path_testing_image.parts
        acquisition_name = Path(path_parts[-1])
        path_acquisition = Path(*path_parts[:-1])

        # Get type of model we are using
        selected_model = path_model.name

        img_name_original = acquisition_name.stem

        # Get the model's native resolution
        model_resolution, patch_size = get_model_native_resolution_and_patch(path_model)

        # Check that the resampled image will be of sufficient size, and if not throw an error.
        im = Image.open(path_testing_image)
        w, h = im.size

        w_resampled = w*(acquired_resolution*zoom_factor)/model_resolution
        h_resampled = h*(acquired_resolution*zoom_factor)/model_resolution

        if w_resampled < patch_size or h_resampled < patch_size:
            if w<=h:
                minimum_zoom_factor = patch_size*model_resolution/(w*acquired_resolution)
            else:
                minimum_zoom_factor = patch_size*model_resolution/(h*acquired_resolution)

            # Round to 1 decimal, always up.
            minimum_zoom_factor = ceil(minimum_zoom_factor*10)/10

            print("ERROR: Due to your given image size, resolution, and zoom factor, the resampled image is smaller than",
                   "the patch size during segmentation. To resolve this, please set a zoom factor greater than ",
                   str(minimum_zoom_factor), ".",
                   "To do this on the command line, call the segmentation with the -z flag, i.e. ",
                   "-z ", str(minimum_zoom_factor),
            )
            sys.exit(4)

        # Performing the segmentation
        axon_segmentation(path_acquisitions_folders=path_acquisition,
                          acquisitions_filenames=[str(path_acquisition / acquisition_name)],
                          path_model_folder=path_model, overlap_value=overlap_value,
                          acquired_resolution=acquired_resolution*zoom_factor)

        if verbosity_level >= 1:
            print(("Image {0} segmented.".format(path_testing_image)))


    else:
        print(("The path {0} does not exist.".format(path_testing_image)))

    return None

def segment_folders(path_testing_images_folder, path_model,
                    overlap_value, 
                    acquired_resolution = None,
                    zoom_factor = 1.0,
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
    :param verbosity_level: Level of verbosity. The higher, the more information is given about the segmentation
    process.
    :return: Nothing.
    '''

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
        print(path_testing_images_folder / file_)
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

            print("ERROR: Due to your given image size, resolution, and zoom factor, at least one image, ", 
                   str(path_testing_images_folder / file_),
                   ", is smaller than",
                   "the patch size after it is resampled during segmentation. To resolve this, please set a zoom factor greater than ",
                   str(minimum_zoom_factor), ".",
                   "To do this on the command line, call the segmentation with the -z flag, i.e. ",
                   "-z ", str(minimum_zoom_factor),
            )
            sys.exit(4)

        axon_segmentation(path_acquisitions_folders=path_testing_images_folder,
                          acquisitions_filenames=[str(path_testing_images_folder  / acquisition_name)],
                          path_model_folder=path_model, overlap_value=overlap_value,
                          acquired_resolution=acquired_resolution*zoom_factor)
        if verbosity_level >= 1:
            tqdm.write("Image {0} segmented.".format(str(path_testing_images_folder / file_)))



    return None

# Main loop

def main(argv=None):

    '''
    Main loop.
    :return: Exit code.
        0: Success
        2: Invalid argument value
        3: Missing value or file
    '''
    print(('AxonDeepSeg v.{}'.format(AxonDeepSeg.__version__)))
    ap = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    requiredName = ap.add_argument_group('required arguments')

    # Setting the arguments of the segmentation
    requiredName.add_argument('-t', '--type', required=True, choices=['SEM', 'TEM', 'BF'], help='Type of acquisition to segment. \n'+
                                                                                        'SEM: scanning electron microscopy samples. \n'+
                                                                                        'TEM: transmission electron microscopy samples. \n' +
                                                                                        'BF: bright-field microscopy samples.')
    requiredName.add_argument('-i', '--imgpath', required=True, nargs='+', help='Path to the image to segment or path to the folder \n'+
                                                                                'where the image(s) to segment is/are located.')

    ap.add_argument("-m", "--model", required=False, help='Folder where the model is located, if different from the default model.')
    ap.add_argument('-s', '--sizepixel', required=False, help='Pixel size of the image(s) to segment, in micrometers. \n'+
                                                              'If no pixel size is specified, a pixel_size_in_micrometer.txt \n'+
                                                              'file needs to be added to the image folder path. The pixel size \n'+
                                                              'in that file will be used for the segmentation.',
                                                              default=None)
    ap.add_argument('-v', '--verbose', required=False, type=int, choices=list(range(0,2)), help='Verbosity level. \n'+
                                                            '0 (default) : Quiet mode. Shows minimal information on the terminal. \n'+
                                                            '1: Developer mode. Shows more information on the terminal, useful for debugging.',
                                                            default=0)
    ap.add_argument('--overlap', required=False, type=int, help='Overlap value (in pixels) of the patches when doing the segmentation. \n'+
                                                            'Higher values of overlap can improve the segmentation at patch borders, \n'+
                                                            'but also increase the segmentation time. \n'+
                                                            'Default value: '+str(default_overlap)+'\n'+
                                                            'Recommended range of values: [10-100]. \n',
                                                            default=default_overlap)
    ap.add_argument("-z", "--zoom", required=False, help='Zoom factor. \n'+
                                                            'When applying the model, the pixel size of the image will be \n'+
                                                            'multiplied by this number.',
                                                            default=None)
    ap._action_groups.reverse()

    # Processing the arguments
    args = vars(ap.parse_args(argv))
    type_ = str(args["type"])
    verbosity_level = int(args["verbose"])
    overlap_value = [int(args["overlap"]), int(args["overlap"])]
    if args["sizepixel"] is not None:
        psm = float(args["sizepixel"])
    else:
        psm = None
    path_target_list = [Path(p) for p in args["imgpath"]]
    new_path = Path(args["model"]) if args["model"] else None 
    if args["zoom"] is not None:
        zoom_factor = float(args["zoom"])
    else:
        zoom_factor = 1.0

    # Preparing the arguments to axon_segmentation function
    path_model= generate_default_parameters(type_, new_path)

    # Tuple of valid file extensions
    validExtensions = (
                        ".jpeg",
                        ".jpg",
                        ".tif",
                        ".tiff",
                        ".png"
                        )

    # Going through all paths passed into arguments
    for current_path_target in path_target_list:

        if not current_path_target.is_dir():
            print(current_path_target)
            if current_path_target.suffix.lower() in validExtensions:

                # Handle cases if no resolution is provided on the CLI
                if psm == None:

                    # Check if a pixel size file exists, if so read it.
                    if (current_path_target.parent / 'pixel_size_in_micrometer.txt').exists():

                        resolution_file = open(current_path_target.parent / 'pixel_size_in_micrometer.txt', 'r')

                        psm = float(resolution_file.read())


                    else:

                        print("ERROR: No pixel size is provided, and there is no pixel_size_in_micrometer.txt file in image folder. ",
                                      "Please provide a pixel size (using argument -s), or add a pixel_size_in_micrometer.txt file ",
                                      "containing the pixel size value."
                        )
                        sys.exit(3)

                # Check that the resampled resolution results in an image large enough for patching


                # Performing the segmentation over the image
                segment_image(
                    path_testing_image=current_path_target,
                    path_model=path_model,
                    overlap_value=overlap_value,
                    acquired_resolution=psm,
                    zoom_factor=zoom_factor,
                    verbosity_level=verbosity_level
                    )

                print("Segmentation finished.")

            else:
                print("The path(s) specified is/are not image(s). Please update the input path(s) and try again.")
                break

        else:

            # Handle cases if no resolution is provided on the CLI
            if psm == None:

                # Check if a pixel size file exists, if so read it.
                if (current_path_target / 'pixel_size_in_micrometer.txt').exists():

                    resolution_file = open(current_path_target / 'pixel_size_in_micrometer.txt', 'r')

                    psm = float(resolution_file.read())

                else:

                    print("ERROR: No pixel size is provided, and there is no pixel_size_in_micrometer.txt file in image folder. ",
                                  "Please provide a pixel size (using argument -s), or add a pixel_size_in_micrometer.txt file ",
                                  "containing the pixel size value."
                    )
                    sys.exit(3)

            # Performing the segmentation over all folders in the specified folder containing acquisitions to segment.
            segment_folders(
                path_testing_images_folder=current_path_target,
                path_model=path_model,
                overlap_value=overlap_value,
                acquired_resolution=psm,
                zoom_factor=zoom_factor,
                verbosity_level=verbosity_level
                )

            print("Segmentation finished.")

    sys.exit(0)

# Calling the script
if __name__ == '__main__':
    main()
