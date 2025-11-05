import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.params import imagej_roi_suffix
from loguru import logger

import os,sys
from pathlib import Path

from read_roi import read_roi_file
from skimage.draw import polygon
import numpy as np

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import argparse
from argparse import RawTextHelpFormatter

def roi_to_mask(roi_folder, image_path):
    """
    Convert a folder of ROI files into a binary mask.
    """
    roi_folder = ads.convert_path(roi_folder)
    image_path = ads.convert_path(image_path)

    ref_img = ads.imread(image_path)
    mask = np.zeros_like(ref_img)

    for fname in os.listdir(roi_folder):
        if fname.lower().endswith(".roi"):
            roi_path = roi_folder / fname

            # Add debug logging
            logger.info(f"Processing {roi_path}")
            
            try:
                rois = read_roi_file(str(roi_path))
                logger.info(f"read_roi_file returned: {type(rois)}")
                if rois is not None:
                    logger.info(f"Number of ROIs in file: {len(rois)}")
            except Exception as e:
                logger.error(f"Error reading {roi_path}: {e}")
                continue

            if rois is None:
                logger.error(f"Can't parse {roi_path} (read_roi_file returned None)")
                continue

            for _, coords in rois.items():
                if 'x' in coords and 'y' in coords:
                    rr, cc = polygon(coords['y'], coords['x'], mask.shape)
                    mask[rr, cc] = 1

    mask_np = np.array(mask)

    output_path = image_path.parent / (image_path.stem + imagej_roi_suffix)
    ads.imwrite(output_path, mask_np * 255)

    logger.info(f"Saved combined mask as {output_path}")

def main(argv=None):
    '''
    Main loop for converting ImageJ ROI files to binary masks.
    :return: Exit code.
        0: Success
        1: Invalid input
        2: Processing error
    '''
    logger.add("axondeepseg_roi.log", level='DEBUG', enqueue=True)

    ap = argparse.ArgumentParser(
        description='Convert ImageJ ROI files to binary masks',
        formatter_class=RawTextHelpFormatter
    )

    requiredName = ap.add_argument_group('required arguments')

    requiredName.add_argument(
        '-i', '--image', 
        required=True, 
        help='Path to the reference image file.',
    )
    requiredName.add_argument(
        '-r', '--roi-folder', 
        required=True, 
        help='Path to the folder containing ROI files.',
    )

    ap._action_groups.reverse()

    args = vars(ap.parse_args(argv))
    
    # Load log file without logger to write
    with open("axondeepseg_roi.log", "a") as f:
        f.write("===================================================================================\n")

    logger.info("AxonDeepSeg ROI to Mask Converter")
    logger.info(f"Command line arguments: {args}")

    image_path = Path(args["image"])
    roi_folder = Path(args["roi_folder"])

    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)

    if not roi_folder.exists():
        logger.error(f"ROI folder not found: {roi_folder}")
        sys.exit(1)

    if not roi_folder.is_dir():
        logger.error(f"ROI path is not a directory: {roi_folder}")
        sys.exit(1)

    roi_files = [f for f in roi_folder.iterdir() if f.suffix.lower() == '.roi']
    if not roi_files:
        logger.warning(f"No .roi files found in directory: {roi_folder}")

    output_path = image_path.parent / (image_path.stem + imagej_roi_suffix)
    
    try:
        logger.info(f"Converting ROIs from {roi_folder} to mask using reference image {image_path}")
        roi_to_mask(roi_folder, image_path, output_path)
        logger.info(f"Successfully created mask at: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during ROI to mask conversion: {e}")
        sys.exit(2)

    sys.exit(0)

if __name__ == '__main__':
    with logger.catch():
        main()