# coding: utf-8

from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter

# Scientific modules imports
import numpy as np

# AxonDeepSeg imports
from AxonDeepSeg.morphometrics.compute_morphometrics import *
import AxonDeepSeg.ads_utils as ads


def launch_morphometrics_computation(path_img, path_prediction, axon_shape="cicle"):
    """
    This function is equivalent to the morphometrics_extraction notebook of AxonDeepSeg.
    It automatically performs all steps (computations, savings, displays,...) of the
    morphometrics extraction of a given sample.
    :param path_img: path of the input image (microscopy sample)
    :param path_prediction: path of the segmented image (output of AxonDeepSeg)
    :param axon_shape: str: shape of the axon, can either be either be circle or an ellipse


    :return: none.
    """
    
    # If string, convert to Path objects
    path_img = convert_path(path_img)
    path_prediction = convert_path(path_prediction)

    try:
        # Read image
        img = ads.imread(path_img)

        # Read prediction
        pred = ads.imread(path_prediction)
    except (IOError, OSError) as e:
        print(("launch_morphometrics_computation: " + str(e)))
        raise
    else:

        # Get axon and myelin masks
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        # Get folder path
        path_folder = path_img.parent

        # Compute and save axon morphometrics
        stats_array = get_axon_morphometrics(pred_axon, path_folder, axon_shape=axon_shape)
        save_axon_morphometrics(path_folder, stats_array)

        # Generate and save displays of axon morphometrics
        fig = draw_axon_diameter(img, path_prediction, pred_axon, pred_myelin, axon_shape=axon_shape)
        save_map_of_axon_diameters(path_folder, fig)

        # Compute and save aggregate morphometrics
        aggregate_metrics = get_aggregate_morphometrics(
            pred_axon, pred_myelin, path_folder, axon_shape=axon_shape
        )
        write_aggregate_morphometrics(path_folder, aggregate_metrics)

def main(argv=None):
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

