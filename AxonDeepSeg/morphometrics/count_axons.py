'''
Finds all morphometric files in a folder and counts the number of axons (myelinated and unmyelinated, 
if applicable) in each file. The results are saved to a CSV file.
'''
from pathlib import Path
from tqdm import tqdm
import argparse
from loguru import logger

import pandas as pd
from skimage import measure

from AxonDeepSeg.ads_utils import imread
from AxonDeepSeg.params import generated_file_suffixes


def main():
    parser = argparse.ArgumentParser(description='Count axons (myelinated and unmyelinated, if applicable).')
    parser.add_argument(
        '-i',
        dest='input_dir',
        type=str,
        help='Path to the folder containing all morphometric files.'
    )
    parser.add_argument(
        '-m', '--mask_mode',
        action='store_true',
        default=False,
        help='Toggles mask mode: use masks instead of .xlsx to count axons.'
    )
    parser.add_argument(
        '-o',
        dest='output_name',
        type=str,
        default='axon_counts.csv',
        help='Name of the output file.',
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    out_name = args.output_name
    mask_mode = args.mask_mode
    assert input_dir.exists(), f'Directory {input_dir} does not exist.'

    # find images except for the masks
    additional_suffixes = [
        '_seg-axonmyelin_filtered.png', '_seg-uaxon_filtered.png', 
        '_seg-axon_filtered.png', '_seg-myelin_filtered.png',
        '_seg-nuclei.png', '_seg-process.png', 
    ]
    ignore_suffixes = [str(s) for s in generated_file_suffixes if str(s).endswith('.png')] + additional_suffixes
    ignore_suffixes = tuple(ignore_suffixes)
    inputs = [f for f in input_dir.glob('*.png') if not f.name.endswith(ignore_suffixes)]

    counts = {'image': [], 'axon_count': [], 'uaxon_count': []}
    axon_morph_suffix = '_axon_morphometrics.xlsx'
    uaxon_morph_suffix = '_uaxon_morphometrics.xlsx'

    total_axon_count = 0
    total_uaxon_count = 0
    total_size = 0
    for img in tqdm(inputs):
        if not mask_mode:
            # read morphometric files
            target_axon_file = str(img.with_suffix('')) + axon_morph_suffix
            target_uaxon_file = str(img.with_suffix('')) + uaxon_morph_suffix
            # filtered morphometrics take precedence over unfiltered morphometrics
            filtered_axon_file = str(img.with_suffix('')) + '_axon_morphometrics_filtered.xlsx'
            filtered_uaxon_file = str(img.with_suffix('')) + '_uaxon_morphometrics_filtered.xlsx'
            if Path(filtered_axon_file).exists():
                target_axon_file = filtered_axon_file
            if Path(filtered_uaxon_file).exists():
                target_uaxon_file = filtered_uaxon_file

            axon_count = len(pd.read_excel(target_axon_file))
            uaxon_count = len(pd.read_excel(target_uaxon_file))
        else:
            target_axonmyelin_mask = str(img.with_suffix('')) + '_seg-axonmyelin.png'
            target_uaxon_mask = str(img.with_suffix('')) + '_seg-uaxon.png'
            filtered_axonmyelin_mask = str(img.with_suffix('')) + '_seg-axonmyelin_filtered.png'
            filtered_uaxon_mask = str(img.with_suffix('')) + '_seg-uaxon_filtered.png'
            if Path(filtered_axonmyelin_mask).exists():
                target_axonmyelin_mask = filtered_axonmyelin_mask
            if Path(filtered_uaxon_mask).exists():
                target_uaxon_mask = filtered_uaxon_mask

            axonmyelin = imread(target_axonmyelin_mask) > 200
            uaxon = imread(target_uaxon_mask) > 200
            total_size += axonmyelin.shape[0] * axonmyelin.shape[1]

            # count axons
            axon_objects = measure.regionprops(measure.label(axonmyelin))
            uaxon_objects = measure.regionprops(measure.label(uaxon))
            axon_count = len(axon_objects)
            uaxon_count = len(uaxon_objects)

        # add data
        counts['image'].append(img.stem)
        counts['axon_count'].append(axon_count)
        counts['uaxon_count'].append(uaxon_count)
        total_axon_count += axon_count
        total_uaxon_count += uaxon_count

    # save counts
    df = pd.DataFrame(counts)
    df.to_csv(out_name, index=False)
    logger.info(f'Total myelinated axon count: {total_axon_count}. Total unmyelinated axon count: {total_uaxon_count}')
    if mask_mode:
        logger.info(f'Total area covered is {total_size} pixels.')

    
if __name__ == "__main__":
    with logger.catch():
        main()