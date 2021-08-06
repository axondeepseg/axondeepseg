from pathlib import Path

# AxonDeepSeg imports
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.visualization import merge_masks
from config import axonmyelin_suffix

from ivadomed import inference as imed_inference

def axon_segmentation(
                    path_acquisitions_folders, 
                    acquisitions_filenames,
                    path_model_folder,
                    overlap_value=[48,48],
                    acquired_resolution=None,
                    segmentations_filenames=[str(axonmyelin_suffix)],
                    ):

    path_model=path_model_folder
    input_filenames = acquisitions_filenames
    options = {"pixel_size": acquired_resolution, "overlap_2D":overlap_value, "binarize_maxpooling": True}

    nii_lst, target_lst = imed_inference.segment_volume(str(path_model), input_filenames, options=options)

    imed_inference.pred_to_png(nii_lst, target_lst, str(Path(input_filenames[0]).parent / Path(input_filenames[0]).stem))

    merge_masks(Path(path_acquisitions_folders) / 'image_seg-axon-manual_pred.png', Path(path_acquisitions_folders) / 'image_seg-myelin-manual_pred.png', Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem) + str(axonmyelin_suffix)))
