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
            case {'axon-diam-gt': threshold}:
                if threshold:
                    df = df[df['axon_diam (um)'] > threshold]
            case _:
                logger.warning(f'Unknown rule: {rule}')
    return df

QA_FLAG_CODES = ('good', 'needs_correction', 'delete')
QA_KEEP_OPTIONS = ('not-deleted', 'good-only')


def read_qa_flags(flags_path):
    '''
    Read the flags exported from the "Single Axon Viewer" tab of the QA report.

    Parameters
    ----------
    flags_path : pathlib.Path
        Path to the CSV file exported by the QA report ("Export flags (CSV)").

    Returns
    -------
    pandas.DataFrame
        One row per flagged axon, with at least the columns "Axon index" (row
        index in the morphometrics file) and "Flag" (good, needs_correction or delete).
    '''
    flags_path = Path(flags_path)
    if not flags_path.exists():
        raise FileNotFoundError(f'QA flags file {flags_path} does not exist.')

    flags = pd.read_csv(flags_path)
    if not {'Axon index', 'Flag'}.issubset(flags.columns):
        raise ValueError(
            f'{flags_path} is not a QA flags export (expected "Axon index" and "Flag" columns). '
            'Files exported by older QA reports ("Axon ID" column) are not supported because their '
            'IDs can be shifted; export the flags again from a regenerated report.'
        )

    flags['Flag'] = flags['Flag'].astype(str).str.strip()
    unknown = set(flags['Flag']) - set(QA_FLAG_CODES)
    if unknown:
        raise ValueError(f'Unknown flag value(s) in {flags_path}: {sorted(unknown)}. Expected one of {QA_FLAG_CODES}.')

    flags['Axon index'] = flags['Axon index'].astype(int)
    if flags['Axon index'].duplicated().any():
        raise ValueError(f'{flags_path} contains the same axon more than once.')

    return flags


def apply_qa_flags(df, flags, keep='not-deleted'):
    '''
    Remove axons from a morphometrics dataframe according to QA report flags.

    Parameters
    ----------
    df : pandas.DataFrame
        Morphometrics of one image. Its first column must hold the axon index
        (the row index written by axondeepseg_morphometrics).
    flags : pandas.DataFrame
        Flags returned by read_qa_flags.
    keep : str
        'not-deleted' keeps every axon except those flagged "delete";
        'good-only' keeps only the axons flagged "good".

    Returns
    -------
    pandas.DataFrame
        The filtered morphometrics.
    '''
    if keep not in QA_KEEP_OPTIONS:
        raise ValueError(f'keep must be one of {QA_KEEP_OPTIONS}, got {keep!r}.')

    ids = df.iloc[:, 0].astype(int)
    flagged = flags.set_index('Axon index')

    unknown_ids = sorted(set(flagged.index) - set(ids))
    if unknown_ids:
        raise ValueError(
            f'{len(unknown_ids)} flagged axon(s) are not in the morphometrics file (e.g. {unknown_ids[:5]}). '
            'The flags were probably exported from a report made on different masks.'
        )

    # The export also stores the g-ratio of each axon: use it to check that the
    # flags refer to the same axons as this morphometrics file.
    if 'g-ratio' in flagged.columns and 'gratio' in df.columns:
        current = pd.Series(df['gratio'].to_numpy(), index=ids.to_numpy()).loc[flagged.index]
        exported = pd.to_numeric(flagged['g-ratio'], errors='coerce')
        both_nan = current.isna().to_numpy() & exported.isna().to_numpy()
        close = np.isclose(current.to_numpy(), exported.to_numpy(), atol=1e-3)
        n_mismatch = int((~(close | both_nan)).sum())
        if n_mismatch:
            raise ValueError(
                f'The g-ratio of {n_mismatch} flagged axon(s) does not match the morphometrics file. '
                'The flags were probably exported from a report made on different masks.'
            )

    flag_of = ids.map(flagged['Flag']).fillna('')
    if keep == 'not-deleted':
        kept = flag_of != 'delete'
    else:
        kept = flag_of == 'good'

    counts = flagged['Flag'].value_counts().to_dict()
    logger.info(
        f'QA flags: {counts.get("good", 0)} good, {counts.get("needs_correction", 0)} needs correction, '
        f'{counts.get("delete", 0)} delete. Keeping {int(kept.sum())} of {len(df)} axons ({keep}).'
    )
    return df[kept.to_numpy()]


def apply_unmyelinated_rules(df, rules):
    for rule in rules:
        match rule:
            case {'axon-diam-gt': threshold}:
                if threshold:
                    df = df[df['axon_diam (um)'] > threshold]
            case {'solidity-gt': threshold}:
                if threshold:
                    df = df[df['solidity'] > threshold]
            case {'axon-area-lt': threshold}:
                if threshold:
                    df = df[df['axon_area (um^2)'] < threshold]
            case _:
                logger.warning(f'Unknown rule: {rule}')
    return df

def mask_updater(morpho_file, axon_type, filtered_df, overwrite, allow_large_images=False):
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
    allow_large_images : bool
        Allow reading masks exceeding PIL's default decompression bomb pixel limit.
    """

    # first, obtain the instance segmentation
    if axon_type == 'unmyelinated':
        target_mask_file = morpho_file.with_name(morpho_file.name.replace('_uaxon_morphometrics.xlsx', '_seg-uaxon.png'))
        if not target_mask_file.exists():
            logger.warning(f'Unmyelinated axon segmentation mask {target_mask_file} was not found. Skipping mask update.')
            return
        uaxon_pred = imread(target_mask_file, allow_large_images=allow_large_images)
        instance_map = measure.label(uaxon_pred, connectivity=1)

    elif axon_type == 'myelinated':
        if not morpho_file.name.endswith('_axon_morphometrics.xlsx'):
            logger.warning(
                f'{morpho_file.name} does not follow the "<image>_axon_morphometrics.xlsx" naming, so the '
                'matching segmentation masks cannot be found. Skipping mask update.'
            )
            return
        axon_myelin_paths = [
            morpho_file.with_name(morpho_file.name.replace('_axon_morphometrics.xlsx', '_seg-axon.png')),
            morpho_file.with_name(morpho_file.name.replace('_axon_morphometrics.xlsx', '_seg-myelin.png')),
        ]
        if not all(f.exists() for f in axon_myelin_paths):
            logger.warning(f'The axon and myelin segmentation masks were not found. Skipping mask update.')
            return
        im_axon = imread(axon_myelin_paths[0], allow_large_images=allow_large_images)
        im_myelin = imread(axon_myelin_paths[1], allow_large_images=allow_large_images)

        target_instance_map = morpho_file.with_name(morpho_file.name.replace('_axon_morphometrics.xlsx', '_instance-map.png'))
        if target_instance_map.exists():
            instance_map = imread(target_instance_map, use_16bit=True, allow_large_images=allow_large_images)
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

def read_morphometrics(morpho_file):
    morpho_file = Path(morpho_file)
    if morpho_file.suffix.lower() == '.csv':
        return pd.read_csv(morpho_file)
    return pd.read_excel(morpho_file)

def write_morphometrics(df, morpho_file):
    morpho_file = Path(morpho_file)
    if morpho_file.suffix.lower() == '.csv':
        df.to_csv(morpho_file, index=False, na_rep='NaN')
    else:
        df.to_excel(morpho_file, index=False, na_rep='NaN')

def process_morphometric_files(
    morpho_files, rules, axon_type, overwrite, update_masks, allow_large_images=False,
    qa_flags=None, qa_keep='not-deleted',
):
    """
    Process a list of morphometric files by applying the given filtering rules.

    Parameters
    ----------
    morpho_files : tuple of pathlib.Path
        List of morphometric files to process (.xlsx or .csv).
    rules : list of dict
        List of filtering rules to apply.
    axon_type : str
        Type of axon ('myelinated' or 'unmyelinated').
    overwrite : bool
        Whether to overwrite the original files or save the filtered results to new files.
    update_masks : bool
        Whether to update the segmentation masks to reflect the filtered morphometrics.
    allow_large_images : bool
        Allow reading masks exceeding PIL's default decompression bomb pixel limit.
    qa_flags : pandas.DataFrame, optional
        Flags exported from the QA report (see read_qa_flags), applied before the rules.
        Only valid for a single myelinated morphometrics file.
    qa_keep : str
        Which axons to keep given the QA flags: 'not-deleted' or 'good-only'.
    """
    logger.info(f'Filtering {axon_type} axons using rules: {rules}')
    for morpho_file in morpho_files:
        df = read_morphometrics(morpho_file)
        original_count = len(df)
        if qa_flags is not None:
            df = apply_qa_flags(df, qa_flags, keep=qa_keep)
        apply_rules_fn = apply_myelinated_rules if axon_type == 'myelinated' else apply_unmyelinated_rules
        df = apply_rules_fn(df, rules)
        filtered_count = len(df)
        logger.info(f'Filtered {original_count - filtered_count} {axon_type} axons from {morpho_file}.')

        # artifact from xlsx import, remove the first column title if required
        df = df.rename(columns={'Unnamed: 0': ''})

        if overwrite:
            write_morphometrics(df, morpho_file)
            logger.info(f'Overwrote original file: {morpho_file}')
        else:
            new_file = morpho_file.with_name(morpho_file.stem + '_filtered' + morpho_file.suffix)
            write_morphometrics(df, new_file)
            logger.info(f'Saved filtered {axon_type} axons to: {new_file}')

        if update_masks:
            mask_updater(
                morpho_file=morpho_file,
                axon_type=axon_type,
                filtered_df=df,
                overwrite=overwrite,
                allow_large_images=allow_large_images,
            )

def main(argv=None):
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
    ap.add_argument(
        "--qa-flags",
        dest="qa_flags",
        help="CSV of axon flags exported from the QA report (Single Axon Viewer > Export flags). "
             "Applied to a single myelinated morphometrics file (-i must be a file), before the rules."
    )
    ap.add_argument(
        "--qa-keep",
        dest="qa_keep",
        choices=QA_KEEP_OPTIONS,
        default='not-deleted',
        help="With --qa-flags: keep every axon except those flagged for deletion (not-deleted, default), "
             "or only the axons flagged as good (good-only)."
    )
    ap.add_argument(
        "--allow-large-images",
        dest="allow_large_images",
        required=False,
        action='store_true',
        default=False,
        help='Allow reading masks exceeding PIL\'s default decompression bomb limit '
             f'(~89 Mpx). \nUse this for large microscopy acquisitions.',
    )

    args = ap.parse_args(argv)

    # input parsing and validation
    input_path = Path(args.input)
    axon_morpho_files = (input_path,) if input_path.is_file() else tuple(input_path.glob("*_axon_morphometrics.xlsx"))
    logger.info(f'Found {len(axon_morpho_files)} myelinated axon morphometric files to process.')

    uaxon_morpho_files = tuple(Path(str(p).replace('_axon_', '_uaxon_')) for p in axon_morpho_files)
    # files without "_axon_" in their name map onto themselves: they are not unmyelinated files
    uaxon_morpho_files = tuple(
        p for p, src in zip(uaxon_morpho_files, axon_morpho_files) if p != src and p.exists()
    )
    logger.info(f'Found {len(uaxon_morpho_files)} unmyelinated axon morphometric files to process.')

    qa_flags = None
    if args.qa_flags:
        if len(axon_morpho_files) != 1 or not input_path.is_file():
            ap.error('--qa-flags applies to one image: -i must be a single morphometrics file.')
        qa_flags = read_qa_flags(args.qa_flags)

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
        allow_large_images=args.allow_large_images,
        qa_flags=qa_flags,
        qa_keep=args.qa_keep,
    )
    process_morphometric_files(
        uaxon_morpho_files,
        rules['unmyelinated'],
        'unmyelinated',
        args.overwrite,
        args.update_masks,
        allow_large_images=args.allow_large_images,
    )


if __name__ == "__main__":
    with logger.catch():
        main()