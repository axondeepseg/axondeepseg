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

def process_morphometric_files(morpho_files, rules, axon_type, overwrite, apply_rules_fn):
    logger.info(f'Filtering {axon_type} axons using rules: {rules}')
    for morpho_file in morpho_files:
        df = pd.read_excel(morpho_file)
        original_count = len(df)
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
        apply_myelinated_rules,
    )
    process_morphometric_files(
        uaxon_morpho_files,
        rules['unmyelinated'],
        'unmyelinated',
        args.overwrite,
        apply_unmyelinated_rules,
    )


if __name__ == "__main__":
    with logger.catch():
        main()