import AxonDeepSeg.ads_utils as ads
from loguru import logger

import os
from read_roi import read_roi_file
from skimage.draw import polygon
import numpy as np

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def roi_to_mask(roi_folder, image_path, output_path):
    """
    Convert a folder of ROI files into a binary mask.
    """
    roi_folder = ads.convert_path(roi_folder)
    image_path = ads.convert_path(image_path)
    output_path = ads.convert_path(output_path)

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
    ads.imwrite(output_path, mask_np * 255)

    logger.info(f"Saved combined mask as {output_path}")
