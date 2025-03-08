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
│   │   subject1-1
│   │   │   metrics_statistics.xlsx
│   │   │   axon_count_figure.png
|   |   subject1-2
│   │   ...
└───subject2
│   │   ...
---

"""

from pathlib import Path
from loguru import logger
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

import AxonDeepSeg
from AxonDeepSeg.params import (
    morph_suffix,
    agg_dir,
    statistics_file_name,
    axon_count_file_name,
    metrics_names,
)


def create_output_folders(
    subject_folder_name: str, subject_name: str, output_folder: Path = agg_dir
):
    """
    Creates an output directory for a subject and its subdirectory for storing aggregated results

    - subject_folder_name (str): Name of the folder containing subject data
    - subject_name (str): Name of the subject
    - output_folder (Path): Base output directory
    """

    morph_subject_path = Path.joinpath(output_folder, subject_folder_name)
    Path.mkdir(morph_subject_path, exist_ok=True)
    morph_subject_name_path = Path.joinpath(morph_subject_path, subject_name)
    Path.mkdir(morph_subject_name_path, exist_ok=True)

    return morph_subject_name_path


def save_axon_count_plot(
    subject_df: pd.DataFrame,
    morph_subject_name_path: Path,
    labels: list,
    subject_name: str,
    axon_count_file_name: str = axon_count_file_name,
):
    """
    Generates and saves a bar plot of axon diameters per subject

    - subject_df (DataFrame): Data containing axon diameters
    - morph_subject_name_path (Path): Path to save the plot
    - labels (list): Axon diameter range labels
    - subject_name (str): Name of the subject
    - axon_count_file_name (str): Name of the output plot file
    """

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=subject_df, x="axon_bin", y="axon_diam (um)", estimator=len, order=labels
    )
    plt.title(f"Axon diameters for subject {subject_name}")
    plt.xlabel("Axon Diameter (um)")
    plt.ylabel("Count")

    save_path = Path.joinpath(morph_subject_name_path, axon_count_file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def load_morphometrics(morph_file: Path, filters: dict):
    """
    Loads morphometric data from an Excel file and filters it

    - morph_file (Path): Path to the morphometrics file
    - filters (dict): Dictionary containing filtering conditions
    """

    df = pd.read_excel(morph_file)
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


def concat_lists_to_df(
    axon_df: pd.DataFrame, gratio_df: pd.DataFrame, myelin_df: pd.DataFrame
):
    """
    Concatenates metric df into a single df

    - axon_df (DataFrame): Axon diameter data
    - gratio_df (DataFrame): G-ratio data
    - myelin_df (DataFrame): Myelin thickness data
    """
    concated_axon_df = concat_metric_df(axon_df)
    concated_gratio_df = concat_metric_df(gratio_df)
    concated_myelin_df = concat_metric_df(myelin_df)
    return pd.concat([concated_axon_df, concated_gratio_df, concated_myelin_df])


def concat_metric_df(metric_df: pd.DataFrame):
    """
    Concatenates a metric df

    - metric_df (DataFrame): Metric data to be aggregated
    """
    concatenated_df = pd.concat(metric_df, axis=1)
    return concatenated_df.T.groupby(level=0).first().T


def save_statistics_excel(
    metrics_df: pd.DataFrame,
    morph_subject_name_path: str,
    statistics_file_name: str = statistics_file_name,
):
    """
    Saves computed statistics as an Excel file

    - metrics_df (DataFrame): Computed statistics data
    - morph_subject_name_path (str): Path to save the file
    - statistics_file_name (str): Name of the statistics file
    """
    full_path = Path.joinpath(morph_subject_name_path, statistics_file_name)
    metrics_df.to_excel(f"{full_path}", index=True)


def aggregate_subject(subject_df: pd.DataFrame, file_name: str, subject_folder: Path):
    """
    Aggregates morphometric statistics for a subject and generates output files

    - subject_df (DataFrame): Morphometric data of the subject
    - file_name (str): Name of the file being processed
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

    subject_folder_name = subject_folder.name
    subject_name = file_name.replace(str(morph_suffix), "")

    morph_subject_name_path = create_output_folders(subject_folder_name, subject_name)

    save_statistics_excel(metrics_df, morph_subject_name_path)
    save_axon_count_plot(subject_df, morph_subject_name_path, labels, subject_name)

    return metrics_df


def aggregate(subjects: list[Path]):
    """
    Aggregates morphometric data for all subjects

    - subjects (list[Path]): List of subject directories.
    """

    for subject_folder in subjects:
        if subject_folder.is_dir():

            for file_path in subject_folder.glob(f"*{str(morph_suffix)}"):
                df, n_filtered = load_morphometrics(
                    file_path, {"gratio_null": True, "gratio_sup": True}
                )

                file_name = (
                    str(file_path).replace(str(subject_folder), "").replace("/", "")
                )
                aggregate_subject(df, file_name, subject_folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing one subdirectory per subject, each containing one or more morphometrics files.",
    )
    args = ap.parse_args()

    logger.add("axondeepseg.log", level="DEBUG", enqueue=True)
    cwd = Path.cwd()
    logger.info(f'Logging initialized for morphometric aggregation in "{cwd}".')
    logger.info(AxonDeepSeg.__version_string__)
    logger.info(f"Arguments: {args}")

    # get subjects
    subjects = [x for x in Path(args.input_dir).iterdir() if x.is_dir()]
    logger.info(f"Found these subjects: {subjects}.")

    Path.mkdir(agg_dir, exist_ok=True)
    aggregate(subjects)


if __name__ == "__main__":
    # with logger.catch():
    main()
