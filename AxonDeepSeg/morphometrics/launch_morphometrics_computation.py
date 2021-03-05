# coding: utf-8

from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
from matplotlib import image
import sys

# Scientific modules imports
import numpy as np
import pandas as pd

# AxonDeepSeg imports
from AxonDeepSeg.morphometrics.compute_morphometrics import *
import AxonDeepSeg.ads_utils as ads
from config import axon_suffix, axonmyelin_suffix, myelin_suffix


def launch_morphometrics_computation(path_img, path_prediction, axon_shape="cicle"):
    """
    This function is equivalent to the morphometrics_extraction notebook of AxonDeepSeg.
    It automatically performs all steps (computations, savings, displays,...) of the
    morphometrics extraction of a given sample.
    :param path_img: path of the input image (microscopy sample)
    :param path_prediction: path of the segmented image (output of AxonDeepSeg)
    :param axon_shape: str: shape of the axon, can either be circle or ellipse


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
    ap.add_argument('-s', '--sizepixel', required=False, help='Pixel size of the image(s) to compute morphometrics, in micrometers. \n'+
                                                              'If no pixel size is specified, a pixel_size_in_micrometer.txt \n'+
                                                              'file needs to be added to the image folder path. The pixel size \n'+
                                                              'in that file will be used for the morphometrics computation.',
                                                              default=None)

    ap.add_argument('-i', '--imgpath', required=True, nargs='+', help='Path to the image.')

    ap.add_argument('-f', '--filename', required=False,  help='Name of the excel file in which the morphometrics will be stored',
                                                              default="Morphometrics"  )

    ap.add_argument('-a', '--axonshape', required=False, help='Axon shape: circle \n' +
                                                                          '\t    ellipse \n' +
                                                            'For computing morphometrics, axon shape can either be a circle or an ellipse', 
                                                              default = "circle")
    

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

    for current_path_target in path_target_list:
        if current_path_target.suffix.lower() in validExtensions:

            #load the axon mask 
            if (Path(str(current_path_target.with_suffix("")) + str(axon_suffix))).exists():
                pred_axon = image.imread(str(current_path_target.with_suffix("")) + str(axon_suffix))
            else: 
                print("ERROR: Segmented axon mask is not present in the image folder. ",
                                    "Please check that the axon mask is located in the image folder. ",
                                    "If it is not present, perform segmentation of the image first using ADS."
                    )
                sys.exit(3)

            #load myelin mask    
            if (Path(str(current_path_target.with_suffix("")) + str(myelin_suffix))).exists():
                pred_myelin = image.imread(str(current_path_target.with_suffix("")) + str(myelin_suffix))
            else: 
                print("ERROR: Segmented myelin mask is not present in the image folder. ",
                                    "Please check that the myelin mask is located in the image folder. ",
                                    "If it is not present, perform segmentation of the image first using ADS."
                    )
                sys.exit(3)

            if args["sizepixel"] is not None:
                psm = float(args["sizepixel"])
            else: # Handle cases if no resolution is provided on the CLI

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

            x = np.array([], dtype=[
                                        ('x0', 'f4'),
                                        ('y0', 'f4'),
                                        ('gratio','f4'),
                                        ('axon_area','f4'),
                                        ('myelin_area','f4'),
                                        ('axon_diam','f4'),
                                        ('myelin_thickness','f4'),
                                        ('axonmyelin_area','f4'),
                                        ('solidity','f4'),
                                        ('eccentricity','f4'),
                                        ('orientation','f4')
                                    ]
                            )
            
            # Compute statistics
            stats_array = get_axon_morphometrics(im_axon=pred_axon, im_myelin=pred_myelin, pixel_size=psm, axon_shape=axon_shape)

            for stats in stats_array:

                x = np.append(x,
                    np.array(
                        [(
                        stats['x0'],
                        stats['y0'],
                        stats['gratio'],
                        stats['axon_area'],
                        stats['myelin_area'],
                        stats['axon_diam'],
                        stats['myelin_thickness'],
                        stats['axonmyelin_area'],
                        stats['solidity'],
                        stats['eccentricity'],
                        stats['orientation']
                        )],
                        dtype=x.dtype)
                    )

            # save the current contents in the file
            if not (filename.lower().endswith((".xlsx", ".csv"))):  # If the user didn't add the extension, add it here
                if filename.lower().endswith('.xlsx'):
                    filename = filename + ".xlsx"
                else: 
                    filename = filename + '.csv'
            try:
                # Export to excel
                pd.DataFrame(x).to_excel(current_path_target.parent /  filename)
                print(f"Moprhometrics file: {filename} has been saved in the {str(current_path_target.parent.absolute())} directory")
            except IOError:
                print("Cannot save morphometrics  data in file '%s'." % file)

        else: 
                print("The path(s) specified is/are not image(s). Please update the input path(s) and try again.")
                break
        
    sys.exit(0)

# Calling the script
if __name__ == '__main__':
    main()
