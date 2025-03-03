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
import os
import math

from AxonDeepSeg.params import morph_suffix, agg_dir


def plot_something():
    ...
    # make a function per figure or something


def load_morphometrics(morph_file: Path):
    # put post-hoc screening here and return dataframe
    # TODO: Add list as a parameter to specify which elements to filter
    
    df = pd.read_excel(morph_file)
    n_filtered = 0
    
    # get rid of null values outliers
    outliers_nan = df[df['gratio'].isnull()]
    print('\n')
    print(f'removing {len(outliers_nan)} lines to exclude NaNs')
    df = df.drop(outliers_nan.index)
    print(f'are there NaN left in the dataframe? {df.isnull().any().any()}')

    # get rid of g-ratio superior to 1
    outliers = df[df['gratio'] >= 1]
    print(f'\nremoving {len(outliers)} lines to ensure g-ratio superior to 1')
    df = df.drop(outliers.index)

    # Visualization of lines remaining
    subjects = pd.unique(df['subject'])
    lengths = {}
    for sub in subjects:
        nb_axons_left = len(df[df['subject'] == sub])
        lengths[sub] = nb_axons_left
        print(f'Nb of axons left for subject {sub}: {nb_axons_left}')
    
    return df, n_filtered


def get_statistics(metric):
  return {
            "Mean": round(metric.mean(), 2),
            "Standard Deviation": round(metric.std(), 2),
            "Min": round(metric.min(), 2),
            "Max": round(metric.max(), 2),
            "Median": round(metric.median(), 2)
        }

def concat_lists_to_df(axon_df, gratio_df, myelin_df):
    concated_axon_df = concat_metric_df(axon_df)
    concated_gratio_df = concat_metric_df(gratio_df)
    concated_myelin_df = concat_metric_df(myelin_df)
    return pd.concat([concated_axon_df, concated_gratio_df, concated_myelin_df])

def concat_metric_df(metric_df):
    concatenated_df = pd.concat(metric_df, axis=1)
    return concatenated_df.T.groupby(level=0).first().T


def aggregate_subject(subject_df: DataFrame, subject: str):
    ...
    # this returns a dataframe with the aggregated subject data
    # also saves a file with the aggregated data
    # Binning information
    bins = [0.5, 1, 1.25, 1.5, 1.75, math.inf]
    labels = ["0.5-1", "1-1.25", "1.25-1.5", "1.5-1.75", "1.75-"]

    # Add a new column for axon bins
    subject_df['axon_bin'] = pd.cut(subject_df['axon_diam (um)'], bins=bins, labels=labels, right=False)

    all_diameters_stats = []
    all_gratio_stats = []
    all_myelin_thickness_stats = []

    # Compute statistics for each bin
    for label in labels:
        axon_diam = subject_df.loc[subject_df['axon_bin'] == label, 'axon_diam (um)']
        gratios = subject_df.loc[subject_df['axon_bin'] == label, 'gratio']
        myelin_thickness = subject_df.loc[subject_df['axon_bin'] == label, 'myelin_thickness (um)']

        all_diameters_stats.append(pd.DataFrame(get_statistics(axon_diam), index=[label]).T)
        all_gratio_stats.append(pd.DataFrame(get_statistics(gratios), index=[label]).T)
        all_myelin_thickness_stats.append(pd.DataFrame(get_statistics(myelin_thickness), index=[label]).T)

    # Concatenate all metric data into the final DataFrame
    concatenated_df = concat_lists_to_df(all_diameters_stats, all_gratio_stats, all_myelin_thickness_stats)
    
    print(f"Formatted DataFrame:\n{concatenated_df}")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=subject_df, x='axon_bin', y='axon_diam (um)', estimator=len, order=labels)
    plt.title(f"Axon diameters for subject {subject}")
    plt.xlabel("Axon Diameter (um)")
    plt.ylabel("Count")

    plt.table(cellText=concatenated_df.values,
              colLabels=concatenated_df.columns,
              rowLabels=[f"{idx[0]} ({idx[1]})" for idx in concatenated_df.index],
              cellLoc="center", loc="bottom", bbox=[0, -1.2, 1, 0.6])

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.show()

def aggregate(input_dir: Path):
    ...
    # put inter-subject stuff in agg_dir
    
    df = load_morphometrics(input_dir)
    subjects = pd.unique(df['subject'])
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        aggregate_subject(subject_df, subject)
    


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
