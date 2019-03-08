import os
import sys
import pickle

# Scientific modules imports
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.misc import imread
from tabulate import tabulate

# Graphs and plots imports
from sys import platform as _platform

if "pytest" in sys.modules:
    import matplotlib as mpl

    mpl.use("Agg")  # Enforces mpl to not open new plot windows
elif _platform == "darwin":  # Mac OSX
    import matplotlib as mpl

    mpl.use("TkAgg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# AxonDeepSeg imports
import AxonDeepSeg.ads_utils
from ..testing.segmentation_scoring import score_analysis, dice


def visualize_training(path_model, start_visu=0):
    """
    :param path_model: path of the folder with the model parameters .ckpt
    :param start_visu: first iterations can reach extreme values, 
        start_visuset another start than epoch 0
    :return: evolution

    figure(1) represent the evolution of the loss and the accuracy evaluated on
    the validation set along the learning process.
    If the learning began from an initial model, the figure plots first the 
    accuracy and loss evolution from this initial model and then stacks the 
    evolution of the model.
    """

    evolution = retrieve_training_data(path_model)

    fig = Figure()
    FigureCanvas(fig)
    # Drawing the evolution curves

    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(
        evolution["steps"][start_visu:],
        evolution["accuracy"][start_visu:],
        "-",
        label="accuracy",
    )
    ax1.set_ylim(ymin=0)
    ax2.plot(
        evolution["steps"][start_visu:],
        evolution["loss"][start_visu:],
        "-r",
        label="loss",
    )

    # Annotating the graph
    ax1.set_title("Accuracy and loss evolution")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")

    return evolution


def visualize_segmentation(path):
    """
    :param path: path of the folder including the data and the results obtained
        after by the segmentation process.
    :return: no return
    if there is a mask (ground truth) in the folder, 
    scores are calculated: sensitivity, errors and dice
    figure(1) segmentation without mrf
    figure(2) segmentation with mrf
    if there is MyelinSeg.jpg in the folder, myelin and image, myelin and axon
    segmentated, myelin and groundtruth are represented
    """

    def _create_fig_helper(overlayed_img, fig_title):
        """
        Helper function to create a figure
        :param overlayed_img: the image to add on top on image_init
        :param fig_title: the title of the figure
        :return: matplotlib.figure.Figure
        """
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.subplots()
        ax.set_title(fig_title)
        ax.imshow(image_init, cmap="gray")
        ax.imshow(overlayed_img, cmap="hsv", alpha=0.7)
        return fig

    path_img = path + "/image.png"
    mask = False

    if not "results.pkl" in os.listdir(path):
        print("results not present")

    file = open(path + "/results.pkl", "r")
    res = pickle.load(file)

    prediction_mrf = res["prediction_mrf"]
    prediction = res["prediction"]
    image_init = imread(path_img, flatten=False, mode="L")
    predict = np.ma.masked_where(prediction == 0, prediction)
    predict_mrf = np.ma.masked_where(prediction_mrf == 0, prediction_mrf)

    title = "Axon Segmentation (with mrf) mask"
    fig = _create_fig_helper(predict_mrf, title)
    fig.savefig(path + "/fig1.png")

    title = "Axon Segmentation (without mrf) mask"
    fig = _create_fig_helper(predict, title)
    fig.savefig(path + "/fig2.png")

    if "mask.png" in os.listdir(path):
        Mask = True
        path_mask = path + "/mask.png"
        mask = preprocessing.binarize(
            imread(path_mask, flatten=False, mode="L"), threshold=125
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

        file = open(path + "/Report_results.txt", "w")
        file.write(text)
        file.close()

    if "MyelinSeg.jpg" in os.listdir(path):
        path_myelin = path + "/MyelinSeg.jpg"
        myelin = preprocessing.binarize(
            imread(path_myelin, flatten=False, mode="L"), threshold=125
        )
        myelin = np.ma.masked_where(myelin == 0, myelin)

        title = "Myelin Segmentation"
        fig = _create_fig_helper(myelin, title)
        fig.savefig(path + "/fig3.png")

        if Mask:
            # New base image for plotting
            image_init = mask
            # Create figure
            title = "Myelin - GroundTruth"
            fig = _create_fig_helper(myelin, title)
            fig.savefig(path + "/fig4.png")


def retrieve_training_data(path_model, path_model_init=None):
    """
    :param path_model: path of the folder with the model parameters .ckpt
    :param path_model_init: if the model is initialized by another, path of its
        folder
    :return: dictionary {steps, accuracy, loss} describing the evolution over 
        epochs of the performance of the model. Stacks the initial model if needed
    """

    file = open(
        path_model + "/evolution.pkl", "rb"
    )  # training variables : loss, accuracy, epoch
    evolution = pickle.load(file)

    if path_model_init:
        file_init = open(path_model_init + "/evolution.pkl", "rb")
        evolution_init = pickle.load(file_init)
        last_epoch = evolution_init["steps"][-1]

        evolution_merged = (
            {}
        )  # Merging the two plots : learning of the init and learning of the model
        for key in ["steps", "accuracy", "loss"]:
            evolution_merged[key] = evolution_init[key] + evolution[key]

        evolution = evolution_merged

    return evolution


def retrieve_hyperparameters(path_model):

    """

    :param path_model: path of the folder with the model parameters .ckpt
    :return: the dict containing the hyperparameters
    """

    file = open(
        path_model + "/hyperparameters.pkl", "r"
    )  # training variables : loss, accuracy, epoch
    return pickle.load(file)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--path_model", required=True, help="")

    args = vars(ap.parse_args())
    path_model = args["path_model"]

    visualize_training(path_model)
