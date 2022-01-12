from pathlib import Path

# AxonDeepSeg imports
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.visualization.merge_masks import merge_masks
from config import axon_suffix, myelin_suffix, axonmyelin_suffix

from ivadomed import inference as imed_inference

def axon_segmentation(
                    path_acquisitions_folders, 
                    acquisitions_filenames,
                    path_model_folder,
                    overlap_value=[48,48],
                    acquired_resolution=None,
                    verbosity_level = 0
                    ):
    '''
    Segment images using IVADOMED.
    :param path_acquisitions_folders: the directory containing the images to segment.
    :param acquisitions_filenames: filenames of the images to segment.
    :param path_model_folder: path to the folder of the IVADOMED-trained model.
    :param overlap_value: the number of pixels to be used for overlap when doing prediction. Higher value means less
    border effects but more time to perform the segmentation.
    :param acquired_resolution: isotropic pixel size of the acquired images.
    :param verbosity_level: Level of verbosity. The higher, the more information is given about the segmentation
    process.
    :return: Nothing.
    '''

    path_model=path_model_folder
    input_filenames = acquisitions_filenames
    options = {"pixel_size": [acquired_resolution, acquired_resolution], "pixel_size_units": "um", "overlap_2D": overlap_value, "binarize_maxpooling": True}

    # IVADOMED automated segmentation
    nii_lst, target_lst = imed_inference.segment_volume(str(path_model), input_filenames, options=options)
    
    imed_inference.pred_to_png(nii_lst, target_lst, str(Path(input_filenames[0]).parent / Path(input_filenames[0]).stem))
    if verbosity_level >= 1:
        print(Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axonmyelin_suffix)))
    
    merge_masks(
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axon_suffix)),
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(myelin_suffix)), 
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axonmyelin_suffix))
        )
