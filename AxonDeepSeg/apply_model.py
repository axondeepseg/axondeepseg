from pathlib import Path

# AxonDeepSeg imports
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.visualization.merge_masks import merge_masks
from config import axon_suffix, myelin_suffix, axonmyelin_suffix
from loguru import logger
from typing import List

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def axon_segmentation(
                    path_inputs: List[Path],
                    path_model: Path,
                    gpu_id: int=0,
                    verbosity_level: int=0,
                    ):
    '''
    Segment images using nnU-Net.

    Parameters
    ----------
    path_inputs : List[pathlib.Path]
        List of images to segment.
    path_model : pathlib.Path
        Path to the folder of the nnU-Net pretrained model.
    gpu_id : int, optional
        GPU ID to use for cuda acceleration, by default 0.
    verbosity_level : int, optional
        Level of verbosity, by default 0.
    '''

def axon_segmentation_deprecated(
                    path_acquisitions_folders, 
                    acquisitions_filenames,
                    path_model_folder,
                    acquired_resolution,
                    overlap_value=None,
                    no_patch=False,
                    gpu_id=0,
                    verbosity_level=0
                    ):
    '''
    Segment images using IVADOMED.
    :param path_acquisitions_folders: the directory containing the images to segment.
    :param acquisitions_filenames: filenames of the images to segment.
    :param path_model_folder: path to the folder of the IVADOMED-trained model.
    :param overlap_value: the number of pixels to be used for overlap between patches when doing prediction.
    Higher value means less border effects but more time to perform the segmentation.
    :param acquired_resolution: isotropic pixel size of the acquired images.
    :param no_patch: If True, the image is segmented without using patches. Default: False. This parameter supersedes
    the "overlap_value" parameter. This option may not be suitable with large images depending on computer RAM capacity.
    :param gpu_id: Number representing the GPU ID for segmentation if available. Default 0.
    :param verbosity_level: Level of verbosity. The higher, the more information is given about the segmentation
    process.
    :return: Nothing.
    '''

    path_model=path_model_folder
    input_filenames = acquisitions_filenames

    # Fill options dictionary
    options = {"pixel_size": [acquired_resolution, acquired_resolution], "pixel_size_units": "um", "binarize_maxpooling": True}
    if no_patch:
        options["no_patch"] = no_patch
        logger.warning("The 'no-patch' option was selected for segmentation. "\
                       "Please note that it may not be suitable with large images depending on computer RAM capacity.")
    else:
        logger.warning("The 'no-patch' option was not selected for segmentation. "\
                       "Please note that this option could potentially produce better results but may not be suitable "\
                       "with large images depending on computer RAM capacity.")
    if overlap_value:
        # When both no_patch and overlap_value are used, the no_patch option supersedes the overlap_value
        # and a warning will be issued by ivadomed while segmenting without patches.
        options["overlap_2D"] = overlap_value
    elif not no_patch:
        # Default overlap is used only without the no_patch option.
        options["overlap_2D"] = [default_overlap, default_overlap]

    # IVADOMED automated segmentation
    nii_lst, _ = imed_inference.segment_volume(str(path_model), input_filenames, gpu_id=gpu_id, options=options)
    
    target_lst = [str(axon_suffix), str(myelin_suffix)]

    imed_inference.pred_to_png(nii_lst, target_lst, str(Path(input_filenames[0]).parent / Path(input_filenames[0]).stem))
    if verbosity_level >= 1:
        print(Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axonmyelin_suffix)))
    
    merge_masks(
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axon_suffix)),
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(myelin_suffix)), 
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axonmyelin_suffix))
        )
