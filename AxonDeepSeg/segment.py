# Segmentation script
# -------------------
# This script lets the user segment automatically one or many images based on the default segmentation models: SEM or
# TEM.
#
# Maxime Wabartha - 2017-08-30

# Imports

import sys
from pathlib import Path

import json
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm
import pkg_resources

import AxonDeepSeg
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.ads_utils import convert_path

# Global variables
SEM_DEFAULT_MODEL_NAME = "default_SEM_model"
TEM_DEFAULT_MODEL_NAME = "default_TEM_model"

MODELS_PATH = pkg_resources.resource_filename('AxonDeepSeg', 'models')
MODELS_PATH = Path(MODELS_PATH)

default_SEM_path = MODELS_PATH / SEM_DEFAULT_MODEL_NAME
default_TEM_path = MODELS_PATH / TEM_DEFAULT_MODEL_NAME
default_overlap = 25

# Definition of the functions

def segment_image(path_testing_image, path_model,
                  overlap_value, config, resolution_model,
                  acquired_resolution = None, verbosity_level=0):

    '''
    Segment the image located at the path_testing_image location.
    :param path_testing_image: the path of the image to segment.
    :param path_model: where to access the model
    :param overlap_value: the number of pixels to be used for overlap when doing prediction. Higher value means less
    border effects but more time to perform the segmentation.
    :param config: dict containing the configuration of the network
    :param resolution_model: the resolution the model was trained on.
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

        # Read image
        img = ads.imread(str(path_testing_image))

        # Generate tmp file
        fp = open(path_acquisition / '__tmp_segment__.png', 'wb+')

        img_name_original = acquisition_name.stem

        ads.imwrite(fp, img, format='png')

        acquisition_name = Path(fp.name).name
        segmented_image_name = img_name_original + '_seg-axonmyelin' + '.png'

        # Performing the segmentation

        axon_segmentation(path_acquisitions_folders=path_acquisition, acquisitions_filenames=[acquisition_name],
                          path_model_folder=path_model, config_dict=config, ckpt_name='model',
                          inference_batch_size=1, overlap_value=overlap_value,
                          segmentations_filenames=segmented_image_name,
                          resampled_resolutions=resolution_model, verbosity_level=verbosity_level,
                          acquired_resolution=acquired_resolution,
                          prediction_proba_activate=False, write_mode=True)

        if verbosity_level >= 1:
            print(("Image {0} segmented.".format(path_testing_image)))

        # Remove temporary file used for the segmentation
        fp.close()
        (path_acquisition / '__tmp_segment__.png').unlink()

    else:
        print(("The path {0} does not exist.".format(path_testing_image)))

    return None

def segment_folders(path_testing_images_folder, path_model,
                    overlap_value, config, resolution_model,
                    acquired_resolution = None,
                    verbosity_level=0):
    '''
    Segments the images contained in the image folders located in the path_testing_images_folder.
    :param path_testing_images_folder: the folder where all image folders are located (the images to segment are located
    in those image folders)
    :param path_model: where to access the model.
    :param overlap_value: the number of pixels to be used for overlap when doing prediction. Higher value means less
    border effects but more time to perform the segmentation.
    :param config: dict containing the configuration of the network
    :param resolution_model: the resolution the model was trained on.
    :param verbosity_level: Level of verbosity. The higher, the more information is given about the segmentation
    process.
    :return: Nothing.
    '''

    # If string, convert to Path objects
    path_testing_images_folder = convert_path(path_testing_images_folder)
    path_model = convert_path(path_model)

    # Update list of images to segment by selecting only image files (not already segmented or not masks)
    img_files = [file for file in path_testing_images_folder.iterdir() if (file.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'))
                 and (not str(file).endswith(('_seg-axonmyelin.png','_seg-axon.png','_seg-myelin.png','mask.png')))]

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
        minimum_resolution = config["trainingset_patchsize"] * resolution_model / min(image_size)

        if acquired_resolution < minimum_resolution:
            print("EXCEPTION: The size of one of the images ({0}x{1}) is too small for the provided pixel size ({2}).\n".format(height, width, acquired_resolution),
                  "The image size must be at least {0}x{0} after resampling to a resolution of {1} to create standard sized patches.\n".format(config["trainingset_patchsize"], resolution_model),
                  "One of the dimensions of the image has a size of {0} after resampling to that resolution.\n".format(round(acquired_resolution * min(image_size) / resolution_model)),
                  "Image file location: {0}".format(str(path_testing_images_folder / file_))
            )

            sys.exit(2)

        selected_model = path_model.name

        # Read image for conversion
        img = ads.imread(str(path_testing_images_folder / file_))

        # Generate tmpfile for segmentation pipeline
        fp = open(path_testing_images_folder / '__tmp_segment__.png', 'wb+')

        img_name_original = file_.stem

        if selected_model == "default_TEM_model":
            ads.imwrite(fp,255-img, format='png')
        else:
            ads.imwrite(fp,img, format='png')

        acquisition_name = Path(fp.name).name
        segmented_image_name = img_name_original + '_seg-axonmyelin' + '.png'

        axon_segmentation(path_acquisitions_folders=path_testing_images_folder, acquisitions_filenames=[acquisition_name],
                              path_model_folder=path_model, config_dict=config, ckpt_name='model',
                              inference_batch_size=1, overlap_value=overlap_value,
                              segmentations_filenames=[segmented_image_name],
                              acquired_resolution=acquired_resolution,
                              verbosity_level=verbosity_level,
                              resampled_resolutions=resolution_model, prediction_proba_activate=False,
                              write_mode=True)

        if verbosity_level >= 1:
            tqdm.write("Image {0} segmented.".format(str(path_testing_images_folder / file_)))

        # Remove temporary file used for the segmentation
        fp.close()
        (path_testing_images_folder / '__tmp_segment__.png').unlink()

    return None

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

    path_config_file = path_model / 'config_network.json'
    config = generate_config_dict(path_config_file)

    return path_model, config

def generate_config_dict(path_to_config_file):
    '''
    Generates the dictionary version of the configuration file from the path where it is located.

    :param path_to_config: relative path where the file config_network.json is located.
    :return: dict containing the configuration of the network, or None if no configuration file was found at the
    mentioned path.
    '''

    # If string, convert to Path objects
    path_to_config_file = convert_path(path_to_config_file)

    try:
        with open(path_to_config_file, 'r') as fd:
            config_network = json.loads(fd.read())

    except:
        raise ValueError("No configuration file available at this path.")

    return config_network

def generate_resolution(type_acquisition, model_input_size):
    '''
    Generates the resolution to use related to the trained modeL.
    :param type_acquisition: String, "SEM" or "TEM"
    :param model_input_size: String or Int, the size of the input.
    :return: Float, the resolution of the model.
    '''

    dict_size = {
        "SEM":{
            "512":0.1,
            "256":0.2
        },
        "TEM":{
            "512":0.01
        }
    }

    return dict_size[str(type_acquisition)][str(model_input_size)]

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
    requiredName.add_argument('-t', '--type', required=True, choices=['SEM','TEM'], help='Type of acquisition to segment. \n'+
                                                                                        'SEM: scanning electron microscopy samples. \n'+
                                                                                        'TEM: transmission electron microscopy samples. ')
    requiredName.add_argument('-i', '--imgpath', required=True, nargs='+', help='Path to the image to segment or path to the folder \n'+
                                                                                'where the image(s) to segment is/are located.')

    ap.add_argument("-m", "--model", required=False, help='Folder where the model is located. \n'+
                                                          'The default SEM model path is: \n'+str(default_SEM_path)+'\n'+
                                                          'The default TEM model path is: \n'+str(default_TEM_path)+'\n')
    ap.add_argument('-s', '--sizepixel', required=False, help='Pixel size of the image(s) to segment, in micrometers. \n'+
                                                              'If no pixel size is specified, a pixel_size_in_micrometer.txt \n'+
                                                              'file needs to be added to the image folder path. The pixel size \n'+
                                                              'in that file will be used for the segmentation.',
                                                              default=None)
    ap.add_argument('-v', '--verbose', required=False, type=int, choices=list(range(0,4)), help='Verbosity level. \n'+
                                                            '0 (default) : Displays the progress bar for the segmentation. \n'+
                                                            '1: Also displays the path of the image(s) being segmented. \n'+
                                                            '2: Also displays the information about the prediction step \n'+
                                                            '   for the segmentation of current sample. \n'+
                                                            '3: Also displays the patch number being processed in the current sample.',
                                                            default=0)
    ap.add_argument('-o', '--overlap', required=False, type=int, help='Overlap value (in pixels) of the patches when doing the segmentation. \n'+
                                                            'Higher values of overlap can improve the segmentation at patch borders, \n'+
                                                            'but also increase the segmentation time. \n'+
                                                            'Default value: '+str(default_overlap)+'\n'+
                                                            'Recommended range of values: [10-100]. \n',
                                                            default=25)
    ap._action_groups.reverse()

    # Processing the arguments
    args = vars(ap.parse_args(argv))
    type_ = str(args["type"])
    verbosity_level = int(args["verbose"])
    overlap_value = int(args["overlap"])
    if args["sizepixel"] is not None:
        psm = float(args["sizepixel"])
    else:
        psm = None
    path_target_list = [Path(p) for p in args["imgpath"]]
    new_path = Path(args["model"]) if args["model"] else None 

    # Preparing the arguments to axon_segmentation function
    path_model, config = generate_default_parameters(type_, new_path)
    resolution_model = generate_resolution(type_, config["trainingset_patchsize"])

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

                # Check that image size is large enough for given resolution to reach minimum patch size after resizing.

                try:
                    height, width, _ = ads.imread(str(current_path_target)).shape
                except:
                    try:
                        height, width = ads.imread(str(current_path_target)).shape
                    except Exception as e:
                        raise e

                image_size = [height, width]
                minimum_resolution = config["trainingset_patchsize"] * resolution_model / min(image_size)

                if psm < minimum_resolution:
                    print("EXCEPTION: The size of one of the images ({0}x{1}) is too small for the provided pixel size ({2}).\n".format(height, width, psm),
                          "The image size must be at least {0}x{0} after resampling to a resolution of {1} to create standard sized patches.\n".format(config["trainingset_patchsize"], resolution_model),
                          "One of the dimensions of the image has a size of {0} after resampling to that resolution.\n".format(round(psm * min(image_size) / resolution_model)),
                          "Image file location: {0}".format(current_path_target)
                    )

                    sys.exit(2)

                # Performing the segmentation over the image
                segment_image(current_path_target, path_model, overlap_value, config,
                            resolution_model,
                            acquired_resolution=psm,
                            verbosity_level=verbosity_level)

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
            segment_folders(current_path_target, path_model, overlap_value, config,
                        resolution_model,
                            acquired_resolution=psm,
                            verbosity_level=verbosity_level)

            print("Segmentation finished.")

    sys.exit(0)

# Calling the script
if __name__ == '__main__':
    main()
