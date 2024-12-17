
# -*- coding: utf-8 -*-

# Basic integrity test to check is AxonDeepSeg is correctly installed.
# Launches a segmentation in the data_test folder.

from pathlib import Path
import numpy as np
import tempfile
import shutil
import sys
import pytest
from loguru import logger

import argparse

# AxonDeepSeg imports
from AxonDeepSeg.testing.segmentation_scoring import pw_dice
from AxonDeepSeg.apply_model import axon_segmentation
import AxonDeepSeg.ads_utils as ads
import AxonDeepSeg.ads_utils
from AxonDeepSeg.params import axonmyelin_suffix

def integrity_test():
    try:

        # get path of directory where AxonDeepSeg was installed
        dir_path = Path(__file__).resolve().parent

        # input parameters
        path = Path('folder_name') / 'file_name'
        model_name = 'model_seg_generalist_light'
        path_model = dir_path / 'models' / model_name
        path_testing = dir_path.parent / 'test' / '__test_files__' / '__test_demo_files__'
        image = Path("image.png")

        with tempfile.TemporaryDirectory() as tmpdirname:
            path_tmp = Path(tmpdirname)
            shutil.copy(path_testing / image, path_tmp)

            # Launch the axon and myelin segmentation on test image sample provided in the installation
            print('Computing the segmentation of axon and myelin on test image.')
            axon_segmentation([str(path_tmp / image)], path_model)

            # Read the ground truth mask and the obtained segmentation mask
            mask = ads.imread(path_testing / 'mask.png')
            pred = ads.imread(path_tmp / (image.stem + str(axonmyelin_suffix)))

            # Generate separate axon and myelin masks of the segmentation output
            print('Generating axon and myelin segmentation masks and saving.')
            gt_axon = mask > 200 # Generate binary image with the axons for the ground truth (myelin=127, axon=255)
            gt_myelin = np.logical_and(mask >= 50, mask <= 200) # Generate binary image with the myelin for the ground truth (myelin=127, axon=255)
            pred_axon = pred > 200 # Generate binary image with the axons for the segmentation (myelin=127, axon=255)
            pred_myelin = np.logical_and(pred >= 50, pred <= 200) # Generate binary image with the myelin for the segmentation (myelin=127, axon=255)

            # Compute Dice between segmentation and ground truth, for both axon and myelin
            dice_axon = pw_dice(pred_axon, gt_axon)
            dice_myelin = pw_dice(pred_myelin, gt_myelin)

        # If all the commands above are executed without bugs, the installation is done correctly
        print("* * * Integrity test passed. AxonDeepSeg is correctly installed. * * * ")
        return 0

    except IOError:

        # Else, there is a problem in the installation
        print("Integrity test failed... ")
        return -1

def main():
    # argparse CLI arguments, default is to run the integrity test only

    parser = argparse.ArgumentParser(description='Run AxonDeepSeg tests.')

    parser.add_argument('--full', action='store_true', help='Run the the full test suite.'
                        'If not specified, then only the integrity tests will be run.')

    args = parser.parse_args()

    if args.full:
        # Run the full test suite
        sys.exit(pytest.main([Path(__file__).resolve().parent.parent / "test"]))

    else:
        # Run the integrity test
        integrity_test()
    return 0

if __name__ == "__main__":
    with logger.catch():
        main()
