# coding: utf-8

import sys
from pathlib import Path
import pickle

# Scientific modules imports
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tabulate import tabulate

# Graphs and plots imports
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# AxonDeepSeg imports
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.ads_utils import convert_path
from AxonDeepSeg.testing.segmentation_scoring import score_analysis, dice

def visualize_segmentation(path):
    """
    :param path: path of the folder including the data and the results obtained
        after by the segmentation process.
    :return: list of matplotlib.figure.Figure
    if there is a mask (ground truth) in the folder,
    scores are calculated: sensitivity, errors and dice
    figure(1) segmentation without mrf
    figure(2) segmentation with mrf
    if there is MyelinSeg.jpg in the folder, myelin and image, myelin and axon
    segmentated, myelin and groundtruth are represented
    """
    # If string, convert to Path objects
    path = convert_path(path)
    
    figs = []

    def _create_fig_helper(overlayed_img, fig_title):
        """
        Helper function to create a figure
        :param overlayed_img: the image to add on top on image_init
        :param fig_title: str title of the figure
        :return: matplotlib.figure.Figure
        """
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.subplots()
        ax.set_title(fig_title)
        ax.imshow(image_init, cmap="gray")
        ax.imshow(overlayed_img, cmap="hsv", alpha=0.7)
        return fig

    path_img = path / "image.png"
    mask = False
    cur_dir_items = [item.name for item in path]

    if "results.pkl" not in cur_dir_items:
        print("results not present")

    file = open(path + "/results.pkl", "r")
    res = pickle.load(file)

    prediction_mrf = res["prediction_mrf"]
    prediction = res["prediction"]

    image_init = ads.imread(path_img)
    predict = np.ma.masked_where(prediction == 0, prediction)
    predict_mrf = np.ma.masked_where(prediction_mrf == 0, prediction_mrf)

    title = "Axon Segmentation (with mrf) mask"
    fig1 = _create_fig_helper(predict_mrf, title)
    figs.append(fig1)

    title = "Axon Segmentation (without mrf) mask"
    fig2 = _create_fig_helper(predict, title)
    figs.append(fig2)

    if "mask.png" in cur_dir_items:
        Mask = True
        path_mask = path / "mask.png"
        mask = preprocessing.binarize(
            ads.imread(path_mask), threshold=125
        )

        acc = accuracy_score(prediction.reshape(-1, 1), mask.reshape(-1, 1))
        score = score_analysis(image_init, mask, prediction)
        Dice = dice(image_init, mask, prediction)["dice"]
        Dice_mean = Dice.mean()
        acc_mrf = accuracy_score(prediction_mrf.reshape(-1, 1), mask.reshape(-1, 1))
        score_mrf = score_analysis(image_init, mask, prediction_mrf)
        Dice_mrf = dice(image_init, mask, prediction_mrf)["dice"]
        Dice_mrf_mean = Dice_mrf.mean()

        headers = ["MRF", "accuracy", "sensitivity", "precision", "diffusion", "Dice"]
        table = [
            ["False", acc, score[0], score[1], score[2], Dice_mean],
            ["True", acc_mrf, score_mrf[0], score_mrf[1], score_mrf[2], Dice_mrf_mean],
        ]

        subtitle2 = "\n\n---Scores---\n\n"
        scores = tabulate(table, headers)
        text = subtitle2 + scores

        subtitle3 = "\n\n---Dice Percentiles---\n\n"
        headers = ["MRF", "Dice 10th", "50th", "90th"]
        table = [
            [
                "False",
                np.percentile(Dice, 10),
                np.percentile(Dice, 50),
                np.percentile(Dice, 90),
            ],
            [
                "True",
                np.percentile(Dice_mrf, 10),
                np.percentile(Dice_mrf, 50),
                np.percentile(Dice_mrf, 90),
            ],
        ]
        scores_2 = tabulate(table, headers)

        text = text + subtitle3 + subtitle3 + scores_2
        print(text)

        file = open(path / "Report_results.txt", "w")
        file.write(text)
        file.close()

    if "MyelinSeg.jpg" in cur_dir_items:
        path_myelin = path / "MyelinSeg.jpg"
        myelin = preprocessing.binarize(
            ads.imread(path_myelin), threshold=125
        )
        myelin = np.ma.masked_where(myelin == 0, myelin)

        title = "Myelin Segmentation"
        fig3 = _create_fig_helper(myelin, title)
        figs.append(fig3)

        if Mask:
            # New base image for plotting
            image_init = mask
            # Create figure
            title = "Myelin - GroundTruth"
            fig4 = _create_fig_helper(myelin, title)
            figs.append(fig4)
    return figs