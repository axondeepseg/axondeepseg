import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.params import axonmyelin_suffix, axon_suffix, myelin_suffix
from loguru import logger

import os,sys
from pathlib import Path

from read_roi import read_roi_file
from skimage.draw import polygon
from skimage.measure import label, regionprops
import numpy as np

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import argparse
from argparse import RawTextHelpFormatter

def pair_axon_myelin_rois(roi_folder, image_path):
    """
    Pair axon and myelin ROIs based on geometric relationships.
    For each axon ROI, find the myelin ROI that contains it.
    """
    roi_folder = ads.convert_path(roi_folder)
    image_path = ads.convert_path(image_path)

    ref_img = ads.imread(image_path)
    height, width = ref_img.shape
    
    all_rois = []
    
    for fname in os.listdir(roi_folder):
        if fname.lower().endswith(".roi"):
            roi_path = roi_folder / fname
            
            try:
                roi_data = read_roi_file(str(roi_path))
                logger.info(f"Processing {roi_path} - {len(roi_data)} ROIs")
                
                for roi_name, coords in roi_data.items():
                    if 'x' in coords and 'y' in coords:
                        # Convert coordinates to integer arrays
                        x_coords = np.array(coords['x'], dtype=int)
                        y_coords = np.array(coords['y'], dtype=int)
                        
                        # Create a binary mask for this single ROI
                        roi_mask = np.zeros((height, width), dtype=bool)
                        rr, cc = polygon(y_coords, x_coords, roi_mask.shape)
                        roi_mask[rr, cc] = True
                        
                        all_rois.append({
                            'name': f"{fname}_{roi_name}",
                            'x': x_coords,
                            'y': y_coords,
                            'mask': roi_mask,
                            'area': np.sum(roi_mask),
                            'centroid': (np.mean(y_coords), np.mean(x_coords))
                        })
                        
            except Exception as e:
                logger.error(f"Error reading {roi_path}: {e}")
                continue

    axon_myelin_pairs = []
    unpaired_rois = all_rois.copy()
    
    # Sort ROIs by area (smaller ones are more likely to be axons)
    unpaired_rois.sort(key=lambda x: x['area'])
    
    while unpaired_rois:
        # Take the smallest ROI as a candidate axon
        candidate_axon = unpaired_rois.pop(0)
        
        # Find the smallest ROI that contains this candidate
        containing_rois = []
        for roi in unpaired_rois:
            if _is_roi_contained(candidate_axon, roi):
                containing_rois.append(roi)
        
        if len(containing_rois) == 1:
            myelin_roi = containing_rois[0]
            axon_myelin_pairs.append({
                'axon': candidate_axon,
                'myelin': myelin_roi
            })
            unpaired_rois.remove(myelin_roi)
            logger.info(f"Paired axon {candidate_axon['name']} with myelin {myelin_roi['name']}")
        else:
            # If no clear pair found, treat as standalone (axon only)
            logger.warning(f"No clear myelin pair found for {candidate_axon['name']}, treating as axon only")
            axon_myelin_pairs.append({
                'axon': candidate_axon,
                'myelin': None
            })
    
    logger.info(f"Pairing complete: {len(axon_myelin_pairs)} axon-myelin pairs found")
    return axon_myelin_pairs

def _is_roi_contained(inner_roi, outer_roi):
    """
    Check if inner_roi is completely contained within outer_roi.
    """
    from matplotlib.path import Path as MPath
    
    outer_points = np.column_stack([outer_roi['x'], outer_roi['y']])
    outer_path = MPath(outer_points)
    
    centroid_inside = outer_path.contains_point([inner_roi['centroid'][1], inner_roi['centroid'][0]])
    
    area_ratio = inner_roi['area'] / outer_roi['area']
    
    return centroid_inside and area_ratio < 0.8

def roi_to_masks(roi_folder, image_path):
    """
    Convert a folder of ROI files into axon and myelin segmentation masks.
    Pairs axon and myelin ROIs, then creates masks by subtracting axon from myelin.
    """
    roi_folder = ads.convert_path(roi_folder)
    image_path = ads.convert_path(image_path)

    ref_img = ads.imread(image_path)
    axon_mask = np.zeros_like(ref_img, dtype=np.uint8)
    myelin_mask = np.zeros_like(ref_img, dtype=np.uint8)
    
    pairs = pair_axon_myelin_rois(roi_folder, image_path)
    
    for pair in pairs:
        axon_roi = pair['axon']
        myelin_roi = pair['myelin']
        
        rr_axon, cc_axon = polygon(axon_roi['y'], axon_roi['x'], axon_mask.shape)
        axon_mask[rr_axon, cc_axon] = 255
        
        if myelin_roi is not None:
            rr_myelin, cc_myelin = polygon(myelin_roi['y'], myelin_roi['x'], myelin_mask.shape)
            
            temp_myelin = np.zeros_like(myelin_mask, dtype=bool)
            temp_myelin[rr_myelin, cc_myelin] = True
            
            temp_axon = np.zeros_like(axon_mask, dtype=bool)
            temp_axon[rr_axon, cc_axon] = True
            
            myelin_only = temp_myelin & ~temp_axon
            myelin_mask[myelin_only] = 255
    
    combined_mask = np.zeros_like(ref_img, dtype=np.uint8)
    combined_mask[myelin_mask == 255] = 127
    combined_mask[axon_mask == 255] = 255
    
    base_path = image_path.parent / image_path.stem
    combined_output_path = Path(base_path.with_name(str(base_path.name) + str(axonmyelin_suffix)))
    axon_output_path = Path(base_path.with_name(str(base_path.name) + str(axon_suffix)))
    myelin_output_path = Path(base_path.with_name(str(base_path.name) + str(myelin_suffix)))
    
    ads.imwrite(combined_output_path, combined_mask)
    ads.imwrite(axon_output_path, axon_mask)
    ads.imwrite(myelin_output_path, myelin_mask)
    
    logger.info(f"Saved combined mask as {str(combined_output_path)}")
    logger.info(f"Saved axon mask as {str(axon_output_path)}")
    logger.info(f"Saved myelin mask as {str(myelin_output_path)}")
    
    return combined_output_path, axon_output_path, myelin_output_path

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
        description='Convert ImageJ ROI files to axon and myelin segmentation masks',
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

    try:
        logger.info(f"Converting ROIs from {roi_folder} to masks using reference image {image_path}")
        combined_path, axon_path, myelin_path = roi_to_masks(roi_folder, image_path)
        logger.info(f"Successfully created masks:")
        logger.info(f"  Combined: {combined_path}")
        logger.info(f"  Axon: {axon_path}")
        logger.info(f"  Myelin: {myelin_path}")
        
    except Exception as e:
        logger.error(f"Error during ROI to mask conversion: {e}")
        sys.exit(2)

    sys.exit(0)

if __name__ == '__main__':
    with logger.catch():
        main()