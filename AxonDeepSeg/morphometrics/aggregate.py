"""
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

The structure of the output directory morphometrics_agg should look like this:

---
morphometrics_agg
└───subject1
│   │   metrics_statistics.xlsx
│   │   axon_count_figure.png
│   │   ...
└───subject2
│   │   ...
---

"""

from pathlib import Path
from loguru import logger
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm

import AxonDeepSeg
from AxonDeepSeg.params import (
    morph_suffix, axonmyelin_suffix,
    agg_dir,
    morph_agg_suffix,
    binned_statistics_filename,
    axon_count_filename,
    metrics_names,
)
from AxonDeepSeg.morphometrics.compute_morphometrics import save_axon_morphometrics


def save_axon_count_plot(
    subject_df: pd.DataFrame,
    morph_subject_name_path: Path,
    labels: list,
    subject_name: str,
    axon_count_filename: str = axon_count_filename,
):
    """
    Generates and saves a bar plot of axon diameters per subject

    - subject_df (DataFrame): Data containing axon diameters
    - morph_subject_name_path (Path): Path to save the plot
    - labels (list): Axon diameter range labels
    - subject_name (str): Name of the subject
    - axon_count_filename (str): Name of the output plot file
    """

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=subject_df, x="axon_bin", y="axon_diam (um)", estimator=len, order=labels
    )
    plt.title(f"Axon diameters for subject {subject_name}")
    plt.xlabel("Axon Diameter (um)")
    plt.ylabel("Count")

    save_path = Path.joinpath(morph_subject_name_path, axon_count_filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def load_morphometrics(df: pd.DataFrame, filters: dict):
    """
    Loads morphometric data from an Excel file and filters it

    - morph_file (Path): Path to the morphometrics file
    - filters (dict): Dictionary containing filtering conditions
    """
    n_filtered = 0

    # get rid of null values outliers
    if filters["gratio_null"] == True:
        outliers_nan = df[df["gratio"].isnull()]
        df = df.drop(outliers_nan.index)
        n_filtered += len(outliers_nan)

    # get rid of g-ratio superior to 1
    if filters["gratio_sup"] == True:
        outliers = df[df["gratio"] >= 1]
        df = df.drop(outliers.index)
        n_filtered += len(outliers)

    return df, n_filtered

def get_statistics(metric: pd.Series):
    """
    Computes statistics for a given metric

    - metric (Series): Data of the metric analyzed
    """
    return {
        "Mean": round(metric.mean(), 2),
        "Standard Deviation": round(metric.std(), 2),
        "Min": round(metric.min(), 2),
        "Max": round(metric.max(), 2),
        "Median": round(metric.median(), 2),
    }

def save_subject_statistics(
    metrics_df: pd.DataFrame,
    morph_subject_name_path: str,
    binned_statistics_filename: str = binned_statistics_filename,
):
    """
    Saves computed statistics as an Excel file

    - metrics_df (DataFrame): Computed statistics data
    - morph_subject_name_path (str): Path to save the file
    - binned_statistics_filename (str): Name of the statistics file
    """
    full_path = Path.joinpath(morph_subject_name_path, binned_statistics_filename)
    metrics_df.to_excel(f"{full_path}", index=True)

def aggregate_subject(
    subject_df: pd.DataFrame, 
    subject_name: str, 
    subject_folder: Path,
    output_dir: Path
):
    """
    Aggregates morphometric statistics for a subject and generates output files

    - subject_df (DataFrame): Morphometric data of the subject
    - subjects_name (str): Name of the file being processed
    - subject_folder (Path): Path to the subject's folder
    """

    # TODO: Change hard-coded bins into inputed value
    bins = [0.5, 1, 1.25, 1.5, 1.75, math.inf]
    labels = ["0.5-1", "1-1.25", "1.25-1.5", "1.5-1.75", "1.75-"]

    # Add a new column for axon bins
    subject_df["axon_bin"] = pd.cut(
        subject_df["axon_diam (um)"], bins=bins, labels=labels, right=False
    )

    # Calculate statistics for each bin
    metrics_dict = {}
    for metric_name, column in metrics_names:
        metric_stats = []

        for label in labels:
            data = subject_df.loc[subject_df["axon_bin"] == label, column]

            if not data.empty:
                metric_stats.append(pd.DataFrame(get_statistics(data), index=[label]).T)
            else:
                metric_stats.append(
                    pd.DataFrame(
                        {label: [None] * len(labels)},
                        index=["Mean", "Standard Deviation", "Min", "Max", "Median"],
                    )
                )

        metrics_dict[metric_name] = pd.concat(metric_stats, axis=1)

    metrics_df = pd.concat(metrics_dict, axis=0, names=["Metric", "Statistic"])

    morph_subject_path = output_dir / subject_name
    Path.mkdir(morph_subject_path, exist_ok=True)

    save_subject_statistics(metrics_df, morph_subject_path)
    save_axon_count_plot(subject_df, morph_subject_path, labels, subject_name)

    return metrics_df

def aggregate(subjects: list[Path], output_dir: Path):
    """
    Aggregates morphometric data for all subjects

    - subjects (list[Path]): List of subject directories.
    - output_folder (Path): Path to the output directory
    """

    for subject_folder in tqdm(subjects):
        subject_data = []
        logger.info(str(morph_suffix))
        for file_path in subject_folder.glob(f"*{str(morph_suffix)}"):
            df = pd.read_excel(file_path)
            
            if df.empty is False:
                df, _ = load_morphometrics(
                    df,
                    filters={"gratio_null": True, "gratio_sup": True}
                )                
                subject_data.append(df)
            
        subject_df = pd.concat(subject_data, ignore_index=True)
        aggregate_subject(subject_df, subject_folder.name, subject_folder, output_dir)

        # Save the subject-aggregated morphometrics
        fname = output_dir / subject_folder.name / f"{subject_folder.name}_{str(morph_agg_suffix)}"
        save_axon_morphometrics(fname, subject_df)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Directory containing one subdirectory per subject, each containing one or more morphometrics files.",
    )
    args = ap.parse_args(argv)

    logger.add("axondeepseg.log", level="DEBUG", enqueue=True)
    logger.info(f'Logging initialized for morphometric aggregation in "{Path.cwd()}".')
    logger.info(AxonDeepSeg.__version_string__)
    logger.info(f"Arguments: {args}")

    # get subjects
    subjects = [x for x in Path(args.input_dir).iterdir() if x.is_dir()]
    subjects = [s for s in subjects if s.name != agg_dir.name]
    logger.info(f"Found {len(subjects)} subjects.")

    # Check that for each image in the subject folder, there a segmentation file with the suffixe {morph_suffix} and {axonmyelin_suffix}

    for subject in subjects:
        morph_files = list(subject.glob(f"*{str(morph_suffix)}"))
        axonmyelin_files = list(subject.glob(f"*{str(axonmyelin_suffix)}"))
        
        # Assert same number of morphometrics and axonmyelin files, else exit and return
        if len(morph_files) != len(axonmyelin_files):
            logger.warning(
                f"Number of morphometrics files ({len(morph_files)}) does not match the number of axonmyelin files ({len(axonmyelin_files)}) in {subject}. Please generate the morphometrics files for all segmentated images."
            )
            return

        if len(morph_files) == 0:
            logger.warning(f"No morphometrics files found in {subject}. Please generate the morphometrics files first.")
            return

        # If there is an axonmyelin file, check that the morphometrics file exists
        for axonmyelin_file in axonmyelin_files:
            morph_file = subject / axonmyelin_file.name.replace(
                str(axonmyelin_suffix), '_' + str(morph_suffix)
            )
            if not morph_file.exists():
                logger.warning(
                    f"File {morph_file} does not exist. Please generate the morphometrics file first."
                )
                return

    output_folder = Path(args.input_dir) / agg_dir
    Path.mkdir(output_folder, exist_ok=True)
    aggregate(subjects, output_folder)

    sys.exit(0)


if __name__ == "__main__":
    with logger.catch():
        main()
