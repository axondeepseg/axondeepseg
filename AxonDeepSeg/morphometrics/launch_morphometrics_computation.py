# coding: utf-8

from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
from matplotlib import image
import sys
import os
from tqdm import tqdm

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
from config import axon_suffix, myelin_suffix, axonmyelin_suffix, morph_suffix, index_suffix, axonmyelin_index_suffix
from AxonDeepSeg.ads_utils import convert_path
from AxonDeepSeg import postprocessing


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

    # Setting the arguments of the saving the morphometrics in excel file
    ap.add_argument('-s', '--sizepixel', required=False, help='Pixel size of the image(s) to compute morphometrics, in micrometers. \n' +
                                                              'If no pixel size is specified, a pixel_size_in_micrometer.txt \n' +
                                                              'file needs to be added to the image folder path. The pixel size \n' +
                                                              'in that file will be used for the morphometrics computation.',
                                                              default=None)

    ap.add_argument('-i', '--imgpath', required=True, nargs='+', help='Path to the image.')

    ap.add_argument('-f', '--filename', required=False,  help='Name of the excel file in which the morphometrics will be stored',
                                                              default=morph_suffix)
    ap.add_argument('-a', '--axonshape', required=False, help='Axon shape: circle \n' +
                                                              '\t    ellipse \n' +
                                                              'For computing morphometrics, axon shape can either be a circle or an ellipse', 
                                                              default="circle")
    
    # Processing the arguments
    args = vars(ap.parse_args(argv))
    path_target_list = [Path(p) for p in args["imgpath"]]
    filename = str(args["filename"])
    axon_shape = str(args["axonshape"])

    # Tuple of valid file extensions
    validExtensions = (
                        ".jpeg",
                        ".jpg",
                        ".tif",
                        ".tiff",
                        ".png"
                        )
    
    flag_morp_batch = False # True, if batch moprhometrics is to computed else False
    target_list = []        # list of image paths for batch morphometrics computations

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
                print(f"ERROR: Segmented axon mask for image: `{str(current_path_target)}` is not present in the image folder. ",
                            "Please check that the axon mask is located in the image folder. ",
                            "If it is not present, perform segmentation of the image first using ADS.\n"
                    )
                sys.exit(3)

            # load myelin mask    
            if (Path(str(current_path_target.with_suffix("")) + str(myelin_suffix))).exists():
                pred_myelin = image.imread(str(current_path_target.with_suffix("")) + str(myelin_suffix))
            else: 
                print(f"ERROR: Segmented myelin mask for image: `{str(current_path_target)}` is not present in the image folder. ",
                            "Please check that the myelin mask is located in the image folder. ",
                            "If it is not present, perform segmentation of the image first using ADS.\n"
                    )
                sys.exit(3)

            if args["sizepixel"] is not None:
                psm = float(args["sizepixel"])
            else:  # Handle cases if no resolution is provided on the CLI

                # Check if a pixel size file exists, if so read it.
                if (current_path_target.parent / 'pixel_size_in_micrometer.txt').exists():

                    resolution_file = open(current_path_target.parent / 'pixel_size_in_micrometer.txt', 'r')

                    psm = float(resolution_file.read())
                else:

                    print(f"ERROR: No pixel size is provided, and there is no pixel_size_in_micrometer.txt file for image: `{str(current_path_target)}` in the image folder. ",
                                "Please provide a pixel size (using argument -s), or add a pixel_size_in_micrometer.txt file ",
                                "containing the pixel size value.\n"
                        )
                    sys.exit(3)

            x = np.array([], dtype=[
                                        ('x0', 'f4'),
                                        ('y0', 'f4'),
                                        ('gratio', 'f4'),
                                        ('axon_area', 'f4'),
                                        ('axon_perimeter', 'f4'),
                                        ('myelin_area', 'f4'),
                                        ('axon_diam', 'f4'),
                                        ('myelin_thickness', 'f4'),
                                        ('axonmyelin_area', 'f4'),
                                        ('axonmyelin_perimeter', 'f4'),
                                        ('solidity', 'f4'),
                                        ('eccentricity', 'f4'),
                                        ('orientation', 'f4')
                                    ]
                        )
            
            # Compute statistics

            stats_array, index_image_array = get_axon_morphometrics(im_axon=pred_axon, im_myelin=pred_myelin, pixel_size=psm, axon_shape=axon_shape, return_index_image=True)

            for stats in stats_array:

                x = np.append(x, np.array(
                        [(
                            stats['x0'],
                            stats['y0'],
                            stats['gratio'],
                            stats['axon_area'],
                            stats['axon_perimeter'],
                            stats['myelin_area'],
                            stats['axon_diam'],
                            stats['myelin_thickness'],
                            stats['axonmyelin_area'],
                            stats['axonmyelin_perimeter'],
                            stats['solidity'],
                            stats['eccentricity'],
                            stats['orientation']
                        )],
                        dtype=x.dtype)
                    )

            morph_filename = current_path_target.stem + "_" + filename

            # save the current contents in the file
            if not (morph_filename.lower().endswith((".xlsx", ".csv"))):  # If the user didn't add the extension, add it here
                morph_filename = morph_filename + '.xlsx' 
            try:
                # Export to excel
                if morph_filename.endswith('.xlsx'):
                    pd.DataFrame(x).to_excel(current_path_target.parent / morph_filename)
                # Export to csv    
                else: 
                    pd.DataFrame(x).to_csv(current_path_target.parent / morph_filename)

                # Generate the index image
                indexes_outfile = current_path_target.parent /(str(current_path_target.with_suffix("")) +
                                                              str(index_suffix))
                ads.imwrite(indexes_outfile, index_image_array)
                # Generate the colored image
                postprocessing.generate_and_save_colored_image_with_index_numbers(
                    filename=current_path_target.parent /(str(current_path_target.with_suffix("")) +
                                                          str(axonmyelin_index_suffix)),
                    axonmyelin_image_path=str(current_path_target.with_suffix("")) + str(axonmyelin_suffix),
                    index_image_array=index_image_array
                )
                    
                print(f"Morphometrics file: {morph_filename} has been saved in the {str(current_path_target.parent.absolute())} directory")
            except IOError:
                print("Cannot save morphometrics data in file '%s'." % morph_filename)

        else: 
            print("The path(s) specified is/are not image(s). Please update the input path(s) and try again.")
            break    
    sys.exit(0)


# Calling the script
if __name__ == '__main__':
    main()
