'''
This script is used to aggregate morphometric data and compute basic statistics,
such as mean g-ratio per subject. It also computes statistics per axon diameter 
ranges (e.g. mean g-ratio for axons with diameter between 0.5 and 1 um, between 
1 and 2, etc.). It assumes the following input directory structure:

---
input_dir
└───subject1
│   │   img_axon_morphometrics.xlsx
|   |   img2_axon_morphometrics.xlsx
│   │   ...
└───subject2
│   │   ...
---

The script will create an aggregate morphometrics file per subject and a global 
one with all subjects, in a output directory called "morphometrics_agg".
'''

from pathlib import Path
from loguru import logger
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from AxonDeepSeg.params import morph_suffix, agg_dir


def plot_something():
    ...
    # make a function per figure or something


def load_morphometrics(morph_file: Path):
    ...
    # put post-hoc screening here and return dataframe
    df = pd.read_excel(morph_file)
    n_filtered = 0
    
    return df, n_filtered


def aggregate_subject(subject_dir: Path):
    ...
    # this returns a dataframe with the aggregated subject data
    # also saves a file with the aggregated data

def aggregate(input_dir: Path):
    ...
    # put inter-subject stuff in agg_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-i', '--input_dir', 
        type=str, 
        help='Directory containing one subdirectory per subject, each containing one or more morphometrics files.'
    )
    args = ap.parse_args()

    logger.add("axondeepseg.log", level='DEBUG', enqueue=True)
    logger.info(f'Logging initialized for morphometric aggregation in "{Path('.')}".')
    logger.info(AxonDeepSeg.__version_string__)
    logger.info(f'Arguments: {args}')

    # get subjects
    subjects = [x for x in Path(args.input_dir).iterdir() if x.is_dir()]
    logger.info(f'Found these subjects: {subjects}.')


if __name__ == '__main__':
    with logger.catch():
        main()
