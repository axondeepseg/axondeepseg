# Successively segments an image with zoom factors within a given range

from pathlib import Path
from math import ceil
from loguru import logger

import AxonDeepSeg.segment as ads_seg
from AxonDeepSeg import ads_utils
from config import axonmyelin_suffix, axon_suffix, myelin_suffix

def get_minimum_zoom_factor(path_img, path_model, acquired_resolution):
    # Get the model's native resolution
    model_resolution, patch_size = ads_seg.get_model_native_resolution_and_patch(path_model)
    im = ads_utils.imread(path_img)

    minimum_zoom_factor = patch_size * model_resolution / (min(im.shape) * acquired_resolution)
    # Round to 1 decimal, always up.
    minimum_zoom_factor = ceil(minimum_zoom_factor*10)/10

    return minimum_zoom_factor

def sweep(
    path_image,
    path_model,
    overlap_value,
    sweep_range,
    sweep_length,
    acquired_resolution = None,
    no_patch=False
    ):
    """
    Wrapper over segment_image to produce segmentations for zoom factor values within a given range.
    :param path_image:          the path of the image to segment.
    :param path_model:          where to access the model
    :param overlap_value:       the number of pixels to be used for overlap when doing prediction. Higher
                                value means less border effects but longer segmentation time.
    :param sweep_range:         tuple with lower and upper bounds for the zoom factor range
    :param sweep_length:        number of equidistant zoom factor values to sample from the range
    :param acquired_resolution: isotropic pixel resolution of the acquired images.
    :param no_patch:            If True, the image is segmented without using patches. Default: False.
    :return: Nothing.
    """

    lower_bound, upper_bound = sweep_range
    # create new directory to store segmentations
    path_results = Path(path_image).parent.resolve() / f'{Path(path_image).stem}_sweep'
    path_results.mkdir(parents=True, exist_ok=True)
    path_seg_outputs = [
        Path(path_image.parent.resolve()) / (Path(path_image).stem + str(axon_suffix)),
        Path(path_image.parent.resolve()) / (Path(path_image).stem + str(myelin_suffix)),
        Path(path_image.parent.resolve()) / (Path(path_image).stem + str(axonmyelin_suffix)),
    ]

    min_zoom_factor = get_minimum_zoom_factor(path_image, path_model, acquired_resolution)
    invalid_lower_bound = False

    for i in range(sweep_length):
        zoom_factor = lower_bound + i * (upper_bound - lower_bound) / sweep_length
        if zoom_factor <= min_zoom_factor:
            invalid_lower_bound = True
            continue

        ads_seg.segment_image(
            path_image,
            path_model,
            overlap_value,
            acquired_resolution,
            zoom_factor,
            no_patch=no_patch
        )    
        # move and rename segmentations
        for path_seg in path_seg_outputs:
            path = Path(path_seg)
            path.rename(path_results / Path(path.stem + f'_zf-{zoom_factor}.png'))

        logger.info(f"Done with zoom factor {zoom_factor}.")

    if invalid_lower_bound:
        warning_msg = "WARNING: The range specified contained invalid zoom factor values, so the lower bound "\
            f"was adjusted. Less than {sweep_length} zoom factor values were processed."
        logger.warning(warning_msg)


