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
from AxonDeepSeg.params import (
    axon_suffix, myelin_suffix, axonmyelin_suffix,
    index_suffix, axonmyelin_index_suffix,
    morph_suffix, unmyelinated_morph_suffix, instance_suffix, 
    unmyelinated_suffix, unmyelinated_index_suffix
)
from AxonDeepSeg.ads_utils import convert_path
from AxonDeepSeg import postprocessing, params
import AxonDeepSeg


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


def load_mask(current_path_target: Path, semantic_class: str, suffix: str):
    """
    This function loads a maskbased on the provided suffix. Will stop execution 
    with exit code 3 if the mask is not found.

    :param current_path_target: Path of the target image
    :param semantic_class: Semantic class of the mask to load ('axon', 'myelin' or 'unmyelinated axon')
    :param suffix: Suffix to determine which mask to load
    :return: Loaded mask image
    """
    mask_path = Path(str(current_path_target.with_suffix("")) + str(suffix))
    if mask_path.exists():
        return image.imread(str(mask_path))
    else:
        msg = f"ERROR: Segmented {semantic_class} mask for image: `{str(current_path_target)}` " \
            f"is not present in the image folder. Please check that the {semantic_class} mask is " \
            " located in the image folder. If it is not present, perform segmentation of the " \
            "image first using ADS.\n"
        logger.error(msg)
        sys.exit(3)


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
        '-c', '--colorize',
        required=False,
        action='store_true',
        help='To save the instance segmentation image.'
    )
    ap.add_argument(
        '-u', '--unmyelinated',
        required=False,
        action='store_true',
        help='Toggles morphometrics for unmyelinated axons. This will only process masks with \n'
            +f'the "{unmyelinated_suffix}" suffix.'
    )

    # Processing the arguments
    args = vars(ap.parse_args(argv))
    path_target_list = [Path(p) for p in args["imgpath"]]
    filename = str(args["filename"])
    axon_shape = str(args["axonshape"])
    colorization_flag = args["colorize"]
    unmyelinated_mode = args["unmyelinated"]
    if unmyelinated_mode:
        if colorization_flag:
            logger.warning("Colorization not supported for unmyelinated axons. Ignoring the -c flag.")
            colorization_flag = False
    if unmyelinated_mode and filename is str(morph_suffix):
        # change to appropriate unmyelinated axon morphometrics filename
        filename = str(unmyelinated_morph_suffix)

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
    logger.info(AxonDeepSeg.__version_string__)
    logger.info(f'Arguments: {args}')

    for dir_iter in path_target_list:
        if dir_iter.is_dir(): # batch morphometrics
            flag_morp_batch = True
            # identify the images; ignore masks but make sure they are present
            target_list += [
                Path(dir_iter / path_target) 
                for path_target in os.listdir(dir_iter) if (
                    Path(path_target).suffix.lower() in validExtensions 
                    and not path_target.endswith(str(axon_suffix))
                    and not path_target.endswith(str(myelin_suffix))
                    and not path_target.endswith(str(axonmyelin_suffix))
                    and not path_target.endswith(str(unmyelinated_suffix))
                    and (
                        (Path(path_target).stem + str(axonmyelin_suffix)) in os.listdir(dir_iter)
                        or (Path(path_target).stem + str(unmyelinated_suffix)) in os.listdir(dir_iter)
                    )
                )
            ]

    if flag_morp_batch: # If flag_morph_batch = True, set the path_target_list to target_list.
        path_target_list = target_list

    for current_path_target in tqdm(path_target_list):
        if current_path_target.suffix.lower() in validExtensions:

            if unmyelinated_mode:
                # load the unmyelinated axon mask
                pred_uaxon = load_mask(current_path_target, 'unmyelinated axon', unmyelinated_suffix)
            else:
                # load the axon and myelin masks
                pred_axon = load_mask(current_path_target, 'axon', axon_suffix)
                pred_myelin = load_mask(current_path_target, 'myelin', myelin_suffix)

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
                im_axon=pred_uaxon if unmyelinated_mode else pred_axon, 
                im_myelin=None if unmyelinated_mode else pred_myelin, 
                pixel_size=psm, 
                axon_shape=axon_shape, 
                return_index_image=True,
                return_border_info=True,
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

                # Generate the colored image; note that its background image is different in unmyelinated mode
                if not unmyelinated_mode:
                    bg_image_path = str(current_path_target.with_suffix("")) + str(axonmyelin_suffix)
                    index_overlay_fname = outfile_basename + str(axonmyelin_index_suffix)
                else:
                    bg_image_path = str(current_path_target.with_suffix("")) + str(unmyelinated_suffix)
                    index_overlay_fname = outfile_basename + str(unmyelinated_index_suffix)
                postprocessing.generate_and_save_colored_image_with_index_numbers(
                    filename=index_overlay_fname,
                    axonmyelin_image_path=bg_image_path,
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
