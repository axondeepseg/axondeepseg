from pathlib import Path
import os
import numpy as np
import torch
from loguru import logger
from typing import List, Literal, NoReturn

# AxonDeepSeg imports
from AxonDeepSeg.visualization.merge_masks import merge_masks
from AxonDeepSeg import ads_utils
from AxonDeepSeg.params import nnunet_suffix, intensity

os.environ['nnUNet_raw'] = 'UNDEFINED'
os.environ['nnUNet_results'] = 'UNDEFINED'
os.environ['nnUNet_preprocessed'] = 'UNDEFINED'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def get_checkpoint_name(checkpoint_folder_path: Path) -> str:
    '''
    Get the name of the checkpoint file in the given folder, with priority for 
    best validation checkpoint.

    Parameters
    ----------
    checkpoint_folder_path : pathlib.Path
        Path to the folder containing the .pth checkpoint file.

    Returns
    -------
    str
        Name of the checkpoint file, e.g. 'checkpoint_best.pth'.
    '''
    if (checkpoint_folder_path / 'checkpoint_best.pth').exists():
        return 'checkpoint_best.pth'
    elif (checkpoint_folder_path / 'checkpoint_final.pth').exists():
        return 'checkpoint_final.pth'
    else:
        # Return checkpoint with most recent modification time
        checkpoints_namesorted=sorted(checkpoint_folder_path.glob('*.pth'))
        return checkpoints_namesorted[-1].name


def extract_from_nnunet_prediction(pred, pred_path, class_name, class_value) -> str:
    '''
    Extracts the given class from the nnunet raw prediction, saves it in a 
    separate mask and return the path of the extracted mask.

    Parameters
    ----------
    pred : np.ndarray
        The raw prediction from nnunet with values 0, 1, 2, ...
    pred_path : pathlib.Path
        Path to the raw prediction file; We expect its filename to end 
        with '_seg-nnunet.png'
    class_name : str
        Name of the class to extract. e.g. 'axon', 'myelin', etc.
    class_value : int
        Value of the class in the raw prediction.

    Errors
    ------
    ValueError
        If the class value is not found in the raw prediction.
    ValueError
        If the raw nnunet prediction file does not end with '_seg-nnunet.png'.

    Returns
    -------
    new_fname : str
        Path to the extracted class mask saved.
    '''

    pred_path = ads_utils.convert_path(pred_path)

    if not np.any(pred == class_value):
        logger.warning(f'Class value {class_value} not found in the raw prediction.')
    
    if not pred_path.name.endswith(str(nnunet_suffix)):
        raise NameError(f'Raw nnunet pred file does not end with "{nnunet_suffix}".')
    
    extraction = np.zeros_like(pred)
    extraction[pred == class_value] = intensity['binary']
    new_fname = str(pred_path).replace(str(nnunet_suffix), f'_seg-{class_name}.png')
    ads_utils.imwrite(new_fname, extraction)

    return new_fname

def find_folds(
            path_model: Path,
            model_type: Literal['light', 'ensemble']='light',
            ) -> List:
    '''
    For a given model, find the folders containing the folds

    Parameters
    ----------
    path_model : pathlib.Path
        Path to the folder model
    model_type :  Literal['light', 'ensemble'], optional
        Type of model, by default 'light'.       

    Returns
    -------
    List
        List of paths to the folds folders.
    '''
    
    if model_type == 'light':
        folds_avail = ['all']
    else:
        folds_avail = [str(f).split('_')[-1] for f in path_model.glob('fold_*')]

    return folds_avail

def axon_segmentation(
                    path_inputs: List[Path],
                    path_model: Path,
                    model_type: Literal['light', 'ensemble']='light',
                    gpu_id: int=-1,
                    verbosity_level: int=0,
                    ) -> NoReturn:
    '''
    Segment images by applying a nnU-Net pretrained model.

    Parameters
    ----------
    path_inputs : List[pathlib.Path]
        List of images to segment. We assume they all exist and are already in 
        the correct format expected by the model (nb of channels, image format).
    path_model : pathlib.Path
        Path to the folder of the nnU-Net pretrained model. We assume it exists.
    model_type : Literal['light', 'ensemble'], optional
        Type of model, by default 'light'.
    gpu_id : int, optional
        GPU ID to use for cuda acceleration. -1 to use CPU, by default -1.
    verbosity_level : int, optional
        Level of verbosity, by default 0.
    '''
    # find all available folds
    folds_avail = find_folds(path_model, model_type)

    # instantiate predictor
    predictor = nnUNetPredictor(
        perform_everything_on_gpu=True if gpu_id >= 0 else False,
        device=torch.device('cuda', gpu_id) if gpu_id >= 0 else torch.device('cpu'),
    )
    logger.info('Running inference on device: {}'.format(predictor.device))

    # find checkpoint name (identical for all folds)
    chkpt_name = get_checkpoint_name(path_model / f'fold_{folds_avail[0]}')
    # init network architecture and load checkpoint
    predictor.initialize_from_trained_model_folder(
        str(path_model),
        use_folds=folds_avail,
        checkpoint_name=chkpt_name,
    )
    logger.info('Model successfully loaded.')

    # create input list
    input_list = [ [str(p)] for p in path_inputs]
    target_suffix = str(nnunet_suffix.with_suffix(''))
    data_format = predictor.dataset_json['file_ending'] # e.g. '.png'
    output_list = [ str(p).replace(data_format, target_suffix) for p in path_inputs ]

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_list,
        output_folder_or_list_of_truncated_output_files=output_list,
        save_probabilities=False,
        overwrite=True,
    )

    output_structure = predictor.dataset_json['labels']
    output_classes = sorted(list(output_structure.keys()))
    output_classes.remove('background')
    is_axonmyelin_seg = ['axon', 'myelin'] == output_classes

    # nnUNet outputs a single file will all classes mapped to consecutive ints
    for pred_path in output_list:
        fname = pred_path + data_format
        raw_pred = ads_utils.imread(fname)
        new_masks = []

        for c in output_classes:
            class_value = output_structure[c]
            new_fname = extract_from_nnunet_prediction(raw_pred, fname, c, class_value)
            new_masks.append(new_fname)
        logger.info(f'Successfully saved masks for classes: {output_classes}.')

        if is_axonmyelin_seg:
            merge_masks(new_masks[0], new_masks[1])

        Path(fname).unlink()
