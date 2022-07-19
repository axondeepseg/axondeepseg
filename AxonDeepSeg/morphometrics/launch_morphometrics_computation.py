# coding: utf-8

from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
from matplotlib import image
import sys
import os
from tqdm import tqdm
from loguru import logger

# Scientific modules imports
import numpy as np
import pandas as pd

# AxonDeepSeg imports
from AxonDeepSeg.morphometrics.compute_morphometrics import (
    get_axon_morphometrics,
    save_axon_morphometrics,
    draw_axon_diameter,
    save_map_of_axon_diameters,
    get_aggregate_morphometrics,
    write_aggregate_morphometrics
)
import AxonDeepSeg.ads_utils as ads
from config import (
    axon_suffix, myelin_suffix, axonmyelin_suffix,
    index_suffix, axonmyelin_index_suffix,
    morph_suffix, instance_suffix,
)
from AxonDeepSeg.ads_utils import convert_path
from AxonDeepSeg import postprocessing, params


def launch_morphometrics_computation(path_img, path_prediction, axon_shape="circle"):
    """
    This function is equivalent to the morphometrics_extraction notebook of AxonDeepSeg.
    It automatically performs all steps (computations, savings, displays,...) of the
    morphometrics extraction of a given sample.
    :param path_img: path of the input image (microscopy sample)
    :param path_prediction: path of the segmented image (output of AxonDeepSeg)
    :param axon_shape: str: shape of the axon, can either be either be circle or an ellipse.
                            if shape of axon = 'circle', equivalent diameter is the diameter of the axon.
                            if shape of axon = 'ellipse', ellipse minor axis is the diameter of the axon.
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
        stats_dataframe = get_axon_morphometrics(pred_axon, path_folder, axon_shape=axon_shape)
        save_axon_morphometrics(path_folder / "morphometrics.pkl", stats_dataframe)

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

    # Setting the arguments of the saving the morphometrics in excel file
    ap.add_argument(
        '-s', '--sizepixel',
        required=False,
        help='Pixel size of the image(s) to compute morphometrics, in micrometers. \n'
            + 'If no pixel size is specified, a pixel_size_in_micrometer.txt \n'
            + 'file needs to be added to the image folder path. The pixel size \n'
            + 'in that file will be used for the morphometrics computation.',
        default=None
    )
    ap.add_argument(
        '-i', '--imgpath',
        required=True,
        nargs='+',
        help='Path to the image.'
    )
    ap.add_argument(
        '-f', '--filename',
        required=False,
        help='Name of the excel file in which the morphometrics will be stored',
        default=morph_suffix
    )
    ap.add_argument(
        '-a', '--axonshape',
        required=False,
        help='Axon shape: circle \n\t    ellipse \n'
            + 'For computing morphometrics, axon shape can either be a circle or an ellipse',
        default="circle"
    )
    ap.add_argument(
        '-b', '--border-info',
        required=False,
        action='store_true',
        help='Adds a flag indicating if the axonmyelin object touches a border along with the \n'
            +'coordinates of its bounding box.'
    )
    ap.add_argument(
        '-c', '--colorize',
        required=False,
        action='store_true',
        help='To save the instance segmentation image.'
    )

    # Processing the arguments
    args = vars(ap.parse_args(argv))
    path_target_list = [Path(p) for p in args["imgpath"]]
    filename = str(args["filename"])
    axon_shape = str(args["axonshape"])
    border_info_flag = args["border_info"]
    colorization_flag = args["colorize"]

    # Tuple of valid file extensions
    validExtensions = (
                        ".jpeg",
                        ".jpg",
                        ".tif",
                        ".tiff",
                        ".png"
                        )

    flag_morp_batch = False # True, if batch morphometrics is to be computed else False
    target_list = []        # list of image paths for batch morphometrics computations

    logger.add("axondeepseg.log", level='DEBUG', enqueue=True)
    logger.info(f'Logging initialized for morphometrics in "{os.getcwd()}".')

    for dir_iter in path_target_list:
        if dir_iter.is_dir(): # batch morphometrics
            flag_morp_batch = True
            target_list += [Path(dir_iter / path_target) for path_target in os.listdir(dir_iter)  \
                                if Path(path_target).suffix.lower() in validExtensions and not path_target.endswith(str(axon_suffix)) \
                                and not path_target.endswith(str(myelin_suffix)) and not path_target.endswith(str(axonmyelin_suffix)) \
                                and ((Path(path_target).stem + str(axonmyelin_suffix)) in os.listdir(dir_iter))]

    if flag_morp_batch: # If flag_morph_batch = True, set the path_target_list to target_list.
        path_target_list = target_list

    for current_path_target in tqdm(path_target_list):
        if current_path_target.suffix.lower() in validExtensions:

            # load the axon mask
            if (Path(str(current_path_target.with_suffix("")) + str(axon_suffix))).exists():
                pred_axon = image.imread(str(current_path_target.with_suffix("")) + str(axon_suffix))
            else:
                msg = f"ERROR: Segmented axon mask for image: `{str(current_path_target)}` is not present " \
                    "in the image folder. Please check that the axon mask is located in the image folder. " \
                    "If it is not present, perform segmentation of the image first using ADS.\n"
                logger.error(msg)
                sys.exit(3)

            # load myelin mask
            if (Path(str(current_path_target.with_suffix("")) + str(myelin_suffix))).exists():
                pred_myelin = image.imread(str(current_path_target.with_suffix("")) + str(myelin_suffix))
            else:
                msg = f"ERROR: Segmented myelin mask for image: `{str(current_path_target)}` is not present " \
                    "in the image folder. Please check that the myelin mask is located in the image folder. " \
                    "If it is not present, perform segmentation of the image first using ADS.\n"
                logger.error(msg)
                sys.exit(3)

            if args["sizepixel"] is not None:
                psm = float(args["sizepixel"])
            else:  # Handle cases if no resolution is provided on the CLI

                # Check if a pixel size file exists, if so read it.
                if (current_path_target.parent / 'pixel_size_in_micrometer.txt').exists():

                    resolution_file = open(current_path_target.parent / 'pixel_size_in_micrometer.txt', 'r')

                    psm = float(resolution_file.read())
                else:

                    msg = "ERROR: No pixel size is provided, and there is no pixel_size_in_micrometer.txt file for " \
                        f"image: `{str(current_path_target)}` in the image folder. Please provide a pixel size (using " \
                        "argument -s), or add a pixel_size_in_micrometer.txt file containing the pixel size value.\n"
                    logger.error(msg)
                    sys.exit(3)

            # Compute statistics
            morph_output = get_axon_morphometrics(
                im_axon=pred_axon, 
                im_myelin=pred_myelin, 
                pixel_size=psm, 
                axon_shape=axon_shape, 
                return_index_image=True,
                return_border_info=border_info_flag,
                return_instance_seg=colorization_flag
            )
            # unpack the morphometrics output
            stats_dataframe, index_image_array = morph_output[0:2]
            if colorization_flag:
                instance_seg_image = morph_output[2]

            morph_filename = current_path_target.stem + "_" + filename

            # save the current contents in the file
            if not (morph_filename.lower().endswith((".xlsx", ".csv"))):  # If the user didn't add the extension, add it here
                morph_filename = morph_filename + '.xlsx'
            try:
                save_axon_morphometrics(current_path_target.parent / morph_filename, stats_dataframe)

                # Generate the index image
                if str(current_path_target) == str(current_path_target.parts[-1]):
                    outfile_basename = str(current_path_target.parent / str(current_path_target.with_suffix("")))
                else:
                    # in case current_path_target already contains the parent directory
                    outfile_basename = str(current_path_target.with_suffix(""))

                ads.imwrite(outfile_basename + str(index_suffix), index_image_array)
                # Generate the colored image
                postprocessing.generate_and_save_colored_image_with_index_numbers(
                    filename=outfile_basename + str(axonmyelin_index_suffix),
                    axonmyelin_image_path=str(current_path_target.with_suffix("")) + str(axonmyelin_suffix),
                    index_image_array=index_image_array
                )
                
                if colorization_flag:
                    # Save instance segmentation
                    ads.imwrite(outfile_basename + str(instance_suffix), instance_seg_image)

                logger.info("Morphometrics file: {} has been saved in the {} directory",
                    morph_filename,
                    str(current_path_target.parent.absolute()),
                )
            except IOError:
                logger.warning(f"Cannot save morphometrics data or associated index images for file {morph_filename}.")

        else:
            logger.warning("The path(s) specified is/are not image(s). Please update the input path(s) and try again.")
            break
    sys.exit(0)


# Calling the script
if __name__ == '__main__':
    with logger.catch():
        main()
