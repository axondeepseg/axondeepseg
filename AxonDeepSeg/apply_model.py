from pathlib import Path
import os
import numpy as np
import torch
from loguru import logger
from typing import List, Literal, Dict

# AxonDeepSeg imports
from AxonDeepSeg.visualization.get_masks import get_masks
from AxonDeepSeg import ads_utils
from config import axon_suffix, myelin_suffix, axonmyelin_suffix

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def setup_environment_vars():
    '''Sets up dummy env vars so that nnUNet does not complain.'''
    os.environ['nnUNet_raw'] = 'UNDEFINED'
    os.environ['nnUNet_results'] = 'UNDEFINED'
    os.environ['nnUNet_preprocessed'] = 'UNDEFINED'

def axon_segmentation(
                    path_inputs: List[Path],
                    path_model: Path,
                    model_type: Literal['light', 'ensemble']='light',
                    gpu_id: int=-1,
                    verbosity_level: int=0,
                    ):
    '''
    Segment images by applying a nnU-Net pretrained model.

    Parameters
    ----------
    path_inputs : List[pathlib.Path]
        List of images to segment. We assume they all exist.
    path_model : pathlib.Path
        Path to the folder of the nnU-Net pretrained model. We assume it exists.
    model_type : Literal['light', 'ensemble'], optional
        Type of model, by default 'light'.
    gpu_id : int, optional
        GPU ID to use for cuda acceleration. -1 to use CPU, by default -1.
    verbosity_level : int, optional
        Level of verbosity, by default 0.
    '''

    setup_environment_vars()
    # find all available folds
    if model_type == 'light':
        folds_avail = ['all']
    else:
        folds_avail = [int(str(f).split('_')[-1]) for f in path_model.glob('fold_*')]

    # instantiate predictor
    predictor = nnUNetPredictor(
        perform_everything_on_gpu=True if gpu_id < 0 else False,
        device=torch.device('cuda', gpu_id) if gpu_id >= 0 else torch.device('cpu'),
    )
    logger.info('Running inference on device: {}'.format(predictor.device))

    # find checkpoint name (identical for all folds)
    chkpt_name = next((path_model / f'fold_{folds_avail[0]}').glob('*.pth')).name
    # init network architecture and load checkpoint
    predictor.initialize_from_trained_model_folder(
        str(path_model),
        use_folds=folds_avail,
        checkpoint_name=chkpt_name,
    )
    logger.info('Model successfully loaded.')

    # create input list
    input_list = [ [str(p)] for p in path_inputs]
    output_list = [ str(p.with_suffix('')) + '_seg-' for p in path_inputs ]

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_list,
        output_folder_or_list_of_truncated_output_files=output_list,
        save_probabilities=False,
        overwrite=True,
    )

    output_structure = predictor.dataset_json['labels']
    output_classes = list(output_structure.keys())
    output_classes.remove('background')
    is_axonmyelin_seg = ['axon', 'myelin'] == sorted(output_classes) 
    # nnUNet outputs a single file will all classes mapped to consecutive ints
    for pred_path in output_list:
        fname = pred_path + predictor.dataset_json['file_ending']
        if is_axonmyelin_seg:
            rescaled = 127 * ads_utils.imread(fname)
            new_fname = fname.replace('_seg-', '_seg-axonmyelin')
            ads_utils.imwrite(new_fname, rescaled)
            print(rescaled.shape)
            get_masks(new_fname)
        else:
            #TODO: save all classes one by one
            pass        
        Path(fname).unlink()
        #TODO: remove nnUNet JSON output files?


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
