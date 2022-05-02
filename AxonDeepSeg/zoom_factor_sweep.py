# Successively segments an image with zoom factors within a given range

from pathlib import Path
from AxonDeepSeg.segment import segment_image
from config import axonmyelin_suffix, axon_suffix, myelin_suffix


def sweep(
    path_image,
    path_model,
    overlap_value,
    sweep_range,
    sweep_length,
    acquired_resolution = None,
    ):
    """
    Wrapper over segment_image to produce segmentations for zoom factor values within a given range.
    :param path_image:          the path of the image to segment.
    :param path_model:          where to access the model
    :param overlap_value:       the number of pixels to be used for overlap when doing prediction. Higher
                                value means less border effects but longer segmentation time.
    :param sweep_range:         upper and lower bounds of the zoom factor range
    :param sweep_length:        number of equidistant zoom factor values to sample from the range
    :param acquired_resolution: isotropic pixel resolution of the acquired images.
    :return: Nothing.
    """

    lower_bound, upper_bound = sweep_range
    # create new directory to store segmentations
    path_results = Path(path_image).parent.absolute() / 'sweep_results'
    path_results.mkdir(parents=True, exist_ok=True)
    path_seg_outputs = [
        Path(path_image).stem + str(axon_suffix),
        Path(path_image).stem + str(myelin_suffix),
        Path(path_image).stem + str(axonmyelin_suffix),
    ]

    for i in range(sweep_length):
        zoom_factor = lower_bound + i * (upper_bound - lower_bound) / sweep_length
        segment_image(
            path_image,
            path_model,
            overlap_value,
            acquired_resolution,
            zoom_factor,
        )    
        # move and rename segmentations
        for path_seg in path_seg_outputs:
            path = Path(path_seg)
            path.rename(path_results / Path(path.stem + f'_zf-{zoom_factor}.png'))

        print(f"Done with zoom factor {zoom_factor}.")