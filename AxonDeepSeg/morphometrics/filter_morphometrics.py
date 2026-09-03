"""
Filter morphometric files based on user-defined constraints. For example, one can remove 
all axons with invadid g-ratio values (g-ratio >= 1 or g-ratio <= 0) or all axons with a 
diameter smaller than a given threshold. The segmentation masks can also be updated to 
reflect the filtered morphometrics.
"""

import argparse
from pathlib import Path
from loguru import logger

import pandas as pd
import yaml
import numpy as np
from skimage import measure

from AxonDeepSeg.ads_utils import imread, imwrite
from AxonDeepSeg.morphometrics.compute_morphometrics import get_watershed_segmentation
from AxonDeepSeg.visualization.merge_masks import merge_masks


def read_config(config_path: Path) -> dict:
    '''
    Read the configuration file containing the filtering criteria

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the configuration file

    Returns
    -------
    dict
        Dictionary containing the filtering criteria for myelinated and unmyelinated axons.
    '''
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file {config_path} does not exist.')
    
    with open(str(config_path), 'r') as f:
        config = yaml.safe_load(f)

    if config.keys() != {'myelinated', 'unmyelinated'}:
        raise ValueError(f'Configuration file {config_path} must contain "myelinated" and "unmyelinated" keys.')
    
    return config

def apply_myelinated_rules(df, rules):
    for rule in rules:
        match rule:
            case {'valid-g-ratio-only': True}:
                df = df[(df['gratio'] > 0) & (df['gratio'] < 1)]
                df = df.dropna(subset=['gratio'])
            case {'axon-diam-gt': threshold} if threshold is not None:
                df = df[df['axon_diam (um)'] > threshold]
            case _:
                logger.warning(f'Unknown rule: {rule}')
    return df

def apply_unmyelinated_rules(df, rules):
    for rule in rules:
        match rule:
            case {'axon-diam-gt': threshold} if threshold is not None:
                df = df[df['axon_diam (um)'] > threshold]
            case {'solidity-gt': threshold} if threshold is not None:
                df = df[df['solidity'] > threshold]
            case {'axon-area-lt': threshold} if threshold is not None:
                df = df[df['axon_area (um^2)'] < threshold]
            case _:
                logger.warning(f'Unknown rule: {rule}')
    return df

def mask_updater(morpho_file, axon_type, filtered_df, overwrite):
    """
    Update the segmentation masks to reflect the filtered morphometrics.

    Parameters
    ----------
    morpho_file : pathlib.Path
        Path to the morphometric file.
    axon_type : str
        Type of axon ('myelinated' or 'unmyelinated').
    filtered_df : pandas.DataFrame
        DataFrame containing the filtered morphometrics.
    overwrite : bool
        Whether to overwrite the original segmentation mask or save the updated mask to a new file.
    """

    # first, obtain the instance segmentation
    if axon_type == 'unmyelinated':
        target_mask_file = morpho_file.with_name(morpho_file.name.replace('_uaxon_morphometrics.xlsx', '_seg-uaxon.png'))
        if not target_mask_file.exists():
            logger.warning(f'Unmyelinated axon segmentation mask {target_mask_file} was not found. Skipping mask update.')
            return
        uaxon_pred = imread(target_mask_file)
        instance_map = measure.label(uaxon_pred, connectivity=1)

    elif axon_type == 'myelinated':
        axon_myelin_paths = [
            morpho_file.with_name(morpho_file.name.replace('_axon_morphometrics.xlsx', '_seg-axon.png')),
            morpho_file.with_name(morpho_file.name.replace('_axon_morphometrics.xlsx', '_seg-myelin.png')),
        ]
        if not all(f.exists() for f in axon_myelin_paths):
            logger.warning(f'The axon and myelin segmentation masks were not found. Skipping mask update.')
            return
        im_axon = imread(axon_myelin_paths[0])
        im_myelin = imread(axon_myelin_paths[1])

        target_instance_map = morpho_file.with_name(morpho_file.name.replace('_axon_morphometrics.xlsx', '_instance-map.png'))
        if target_instance_map.exists():
            instance_map = imread(target_instance_map, use_16bit=True)
        else:
            logger.warning(f'Myelinated axon instance map {target_instance_map} was not found. Computing instance map from original segmentation masks instead.')

            # duplication of the code in compute_morphometrics.py to get the instance map from the axon and myelin masks
            im_axon_label = measure.label(im_axon, connectivity=2)
            axon_objects = measure.regionprops(im_axon_label)
            index_centroids = (
                [int(props.centroid[0]) for props in axon_objects],
                [int(props.centroid[1]) for props in axon_objects],
            )
            instance_map = get_watershed_segmentation(im_axon, im_myelin, index_centroids)

    valid_ids = filtered_df.iloc[:, 0].astype(int).values
    valid_ids = [i + 1 for i in valid_ids]
    # create a binary mask for invalid axons
    deletion_mask = np.isin(instance_map, valid_ids, invert=True)

    if axon_type == 'unmyelinated':
        uaxon_pred[deletion_mask] = 0
        if overwrite:
            imwrite(target_mask_file, uaxon_pred)
            logger.info(f'Overwrote unmyelinated axon segmentation mask: {target_mask_file}')
        else:
            new_mask_file = target_mask_file.with_name(target_mask_file.stem + '_filtered.png')
            imwrite(new_mask_file, uaxon_pred)
            logger.info(f'Saved updated unmyelinated axon segmentation mask to: {new_mask_file}')
    elif axon_type == 'myelinated':
        im_axon[deletion_mask] = 0
        im_myelin[deletion_mask] = 0
        if overwrite:
            target_mask_paths = axon_myelin_paths
            for mask_file, mask in zip(axon_myelin_paths, [im_axon, im_myelin]):
                imwrite(mask_file, mask)
                logger.info(f'Overwrote {mask_file}')
        else:
            target_mask_paths = [
                f.with_name(f.stem + '_filtered.png') for f in axon_myelin_paths
            ]
            for new_mask_file, mask in zip(target_mask_paths, [im_axon, im_myelin]):
                imwrite(new_mask_file, mask)
                logger.info(f'Saved updated segmentation mask to: {new_mask_file}')
        # also update the axonmyelin mask
        merge_masks(*target_mask_paths)

def process_morphometric_files(morpho_files, rules, axon_type, overwrite, update_masks):
    """
    Process a list of morphometric files by applying the given filtering rules.

    Parameters
    ----------
    morpho_files : tuple of pathlib.Path
        List of morphometric files to process.
    rules : list of dict
        List of filtering rules to apply.
    axon_type : str
        Type of axon ('myelinated' or 'unmyelinated').
    overwrite : bool
        Whether to overwrite the original files or save the filtered results to new files.
    update_masks : bool
        Whether to update the segmentation masks to reflect the filtered morphometrics.
    """
    logger.info(f'Filtering {axon_type} axons using rules: {rules}')
    for morpho_file in morpho_files:
        df = pd.read_excel(morpho_file)
        original_count = len(df)
        apply_rules_fn = apply_myelinated_rules if axon_type == 'myelinated' else apply_unmyelinated_rules
        df = apply_rules_fn(df, rules)
        filtered_count = len(df)
        logger.info(f'Filtered {original_count - filtered_count} {axon_type} axons from {morpho_file}.')

        # artifact from xlsx import, remove the first column title if required
        df = df.rename(columns={'Unnamed: 0': ''})

        if overwrite:
            df.to_excel(morpho_file, index=False)
            logger.info(f'Overwrote original file: {morpho_file}')
        else:
            new_file = morpho_file.with_name(morpho_file.stem + '_filtered.xlsx')
            df.to_excel(new_file, index=False)
            logger.info(f'Saved filtered {axon_type} axons to: {new_file}')

        if update_masks:
            mask_updater(
                morpho_file=morpho_file,
                axon_type=axon_type,
                filtered_df=df,
                overwrite=overwrite,
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input morphometric file or folder containing morphometric files"
    )
    ap.add_argument(
        "-c", "--config",
        help="Path to the configuration file containing filtering criteria (defaults to AxonDeepSeg/morphometrics/filter.yaml)" 
    )
    ap.add_argument(
        "-m", "--update_masks",
        action="store_true",
        help="Update the segmentation masks to reflect the filtered morphometrics"
    )
    ap.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite the original morphometric files (and segmentation masks if -m is set)"
    )

    args = ap.parse_args()

    # input parsing and validation
    input_path = Path(args.input)
    axon_morpho_files = (input_path,) if input_path.is_file() else tuple(input_path.glob("*_axon_morphometrics.xlsx"))
    logger.info(f'Found {len(axon_morpho_files)} myelinated axon morphometric files to process.')

    uaxon_morpho_files = tuple(Path(str(p).replace('_axon_', '_uaxon_')) for p in axon_morpho_files)
    uaxon_morpho_files = tuple(p for p in uaxon_morpho_files if p.exists())
    logger.info(f'Found {len(uaxon_morpho_files)} unmyelinated axon morphometric files to process.')

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "filter.yaml"
    rules = read_config(config_path)

    process_morphometric_files(
        axon_morpho_files,
        rules['myelinated'],
        'myelinated',
        args.overwrite,
        args.update_masks,
    )
    process_morphometric_files(
        uaxon_morpho_files,
        rules['unmyelinated'],
        'unmyelinated',
        args.overwrite,
        args.update_masks,
    )


if __name__ == "__main__":
    with logger.catch():
        main()