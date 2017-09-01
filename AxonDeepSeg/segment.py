# Segmentation script
# -------------------
# This script lets the user segment automatically one or many images based on the default segmentation models: SEM or
# TEM.
#
# Maxime Wabartha - 2017-08-30

# Imports

from AxonDeepSeg.apply_model import axon_segmentation
import os, json

# Global variables
SEM_DEFAULT_MODEL_NAME = "default_SEM_model_v1"
TEM_DEFAULT_MODEL_NAME = "default_TEM_model_v1"


# Definition of the functions

def segment_folders(type_, path_testing_images_folder, path_model,
                    overlap_value, config, resolution_model, segmented_image_prefix):
    '''
    Segments the images contained in the image folders located in the path_testing_images_folder.
    :param type_: type of the acquisition (SEM, TEM)
    :param path_testing_images_folder: the folder where all image folders are located (the images to segment are located
    in those image folders)
    :param path_model: where to access the model
    :param overlap_value: the number of pixels to be used for overlap when doing prediction. Higher value means less
    border effects but more time to perform the segmentation.
    :param config: dict containing the configuration of the network
    :param resolution_model: the resolution the model was trained on.
    :param segmented_image_prefix: the prefix to add before the segmented image.
    :return:
    '''

    # We loop over all image folders in the specified folded and we segment them one by one.
    for image_folder in os.listdir(path_testing_images_folder):

        path_image_folder = os.path.join(path_testing_images_folder, image_folder)
        if os.path.isdir(path_image_folder):

            # We loop through every file in the folder as we look for an image to segment
            for file_ in os.listdir(path_image_folder):

                # We check if there is an image called image.png, then we segment this one in priority
                if file_ == "image.png":

                    # Performing the segmentation
                    segmented_image_name = segmented_image_prefix + file_
                    axon_segmentation(path_my_datas=path_image_folder, path_model=path_model,
                                      config=config, ckpt_name='model',
                                      batch_size=1, crop_value=overlap_value, imagename=segmented_image_name,
                                      general_pixel_sizes=resolution_model, pred_proba=False, write_mode=True)

                # Else we perform the segmentation for the first image that is a png file and not a segmented image

                elif (file_[-4:] == ".png") and (not (file_[:len(segmented_image_prefix)] == segmented_image_prefix)):

                    # Performing the segmentation
                    segmented_image_name = segmented_image_prefix + file_
                    axon_segmentation(path_my_datas=path_image_folder, config=config, ckpt_name='model',
                                      batch_size=1, crop_value=overlap_value, imagename=segmented_image_name,
                                      general_pixel_sizes=resolution_model, pred_proba=False, write_mode=True)

                    # The segmentation has been done for this image folder, we go to the next one.
    return None

def generate_default_parameters(type_acquisition):
    '''
    Generates the parameters used for segmentation for the default model corresponding to the type_model acquisition.
    :param type_model: The type of model to get the parameters from.
    :return: the config dictionary.
    '''

    # Building the path of the wanted default model
    if type_acquisition == 'SEM':
        path_model = os.path.join('../models/defaults/', SEM_DEFAULT_MODEL_NAME)
    elif type_acquisition == 'TEM':
        path_model = os.path.join('../models/defaults/', TEM_DEFAULT_MODEL_NAME)
    else:
        path_model = '../models/defaults/default_model'

    path_config_file = os.path.join(path_model, 'config_network.json')
    config = generate_config_dict(path_config_file)

    return path_model, config

def generate_config_dict(path_to_config_file):
    '''
    Generates the dictionary version of the configuration file from the path where it is located.

    :param path_to_config: relative path where the file config_network.json is located.
    :return: dict containing the configuration of the network, or None if no configuration file was found at the
    mentioned path.
    '''

    try:
        with open(path_to_config_file, 'r') as fd:
            config_network = json.loads(fd.read())

    except ValueError:
        print "No configuration file available at this path."
        config_network = None

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
            "512":0.1
        },
        "TEM":{
            "512":0.01
        }
    }

    return dict_size[str(type_acquisition)][str(model_input_size)]

# Main loop

def main():
    '''
    Main loop.
    :return: None.
    '''
    import argparse
    ap = argparse.ArgumentParser()

    # Setting the arguments of the segmentation
    ap.add_argument("-t", "--type", required=True, help="Choose the type of acquisition you want to segment.") # type
    ap.add_argument("-p", "--path", required=True, help="Folder where the acquisition folder are located")
    ap.add_argument("-o", "--overlap", required=False, help="Overlap value when doing the segmentation. The higher the"
                                                            "value, the longer it will take to segment the whole image",
                    default=25)

    # Processing the arguments
    args = vars(ap.parse_args())
    type_ = str(args["type"])
    path_testing_images_folder = str(args["path"])
    overlap_value = int(args["overlap"])

    # Preparing the arguments to axon_segmentation function
    path_model, config = generate_default_parameters(type_)
    resolution_model = generate_resolution(type_, config["trainingset_patchsize"])
    segmented_image_prefix = "segmentation_"

    # Performing the segmentation over all folders containing acquisitions in the specified folder
    segment_folders(type_, path_testing_images_folder, path_model, overlap_value, config, resolution_model, segmented_image_prefix)


# Calling the script

if __name__ == '__main__':
    main()