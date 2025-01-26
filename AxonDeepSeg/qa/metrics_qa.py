# coding: utf-8

# Scientific modules imports
import numpy as np
import scipy

# Graphs and plots imports
import matplotlib.pyplot as plt
import pandas as pd

import pathlib
from pathlib import Path

import matplotlib.pyplot as plt

import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

mpl_config = Path(pathlib.Path(__file__).parent.resolve() / 'custom_matplotlibrc')
plt.style.use(mpl_config)
plt.rcParams["figure.figsize"] = (9,6)

class MetricsQA:
    def __init__(self, morphometrics_file):
        """
        :param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
        :return: the pixel size value.
        """

        self.file_name = Path(morphometrics_file)

        self.df = pd.read_csv(morphometrics_file)

    def list_metrics(self):
        print('\n'.join(list(self.df.columns.values[3:])))

    def plot(self, metric_name, save_folder = None, quiet = False):
        x = self.df[metric_name].to_numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1])

        # bins='fd' uses the Freedman Diaconis Estimator to find the optimal number of bins
        count, bins, ignored = ax1.hist(x,bins='fd', histtype='bar', ec='black')
        ax1.set(xlabel=metric_name, ylabel='Count', title='Histogram') 

        ax2.axis('off')
        ax2.set(title='Stats')

        ax2.annotate('mean: ', xy=(0, 0.9))
        ax2.annotate('median: ', xy=(0, 0.8))
        ax2.annotate('std: ', xy=(0, 0.7))
        ax2.annotate('iqr: ', xy=(0, 0.6))
        ax2.annotate('min: ', xy=(0, 0.5))
        ax2.annotate('max: ', xy=(0, 0.4))
        ax2.annotate('NaNs #:', xy=(0, 0.3))

        ax2.annotate(
            np.format_float_positional(np.nanmean(x), precision=2, trim='0'),
            xy=(1, 0.9)
            )
        ax2.annotate(
            np.format_float_positional(np.nanmedian(x), precision=2, trim='0'),
            xy=(1, 0.8)
            )
        ax2.annotate(
            np.format_float_positional(np.nanstd(x), precision=2, trim='0'),
            xy=(1, 0.7)
            )
        ax2.annotate(
            np.format_float_positional(scipy.stats.iqr(x[~np.isnan(x)]), precision=2, trim='0'),
            xy=(1, 0.6))
        
        ax2.annotate(
            np.format_float_positional(np.nanmin(x), precision=2, trim='0'),
            xy=(1, 0.5)
            )
        ax2.annotate(
            np.format_float_positional(np.nanmax(x), precision=2, trim='0'),
            xy=(1, 0.4)
            )
        ax2.annotate(np.sum(np.isnan(x)), xy=(1, 0.3))
        if quiet == False:
            fig.show()

        if save_folder is not None:
            plt.savefig(Path(Path(save_folder) / metric_name))
    
    def plot_all(self, save_folder=None, quiet=False):
        metric_list = list(self.df.columns.values[3:])

        for metric in metric_list:
            self.plot(metric, save_folder, quiet)

    def get_flagged_objects(self, axon_file, myelin_file, pixel_file, folder, save_folder):

        axon_img = ads.imread(axon_file)
        myelin_img = ads.imread(myelin_file)

        morph_out = get_axon_morphometrics(
                axon_img, 
                path_folder=folder, 
                im_myelin=myelin_img, 
                pixel_size=pixel_file, 
                axon_shape="circle", 
                return_index_image=False, 
                return_border_info=False,
                return_instance_seg=True
            )
        df = self.df

        
        flagged_objects = np.array([])
        flagged_objects = np.append(flagged_objects, df.loc[df['gratio'] >=0.99].index.to_numpy())
        flagged_objects = np.append(flagged_objects, df.loc[df['axon_area (um^2)'] <= min(df['axon_area (um^2)'])*5].index.to_numpy())
        flagged_objects = np.append(flagged_objects, df.loc[df['myelin_area (um^2)'] <= min(df['myelin_area (um^2)'])*5].index.to_numpy())

        flagged_objects = np.unique(flagged_objects)

        mask = np.zeros_like(morph_out[2])

        # go through each element in arr
        for id in flagged_objects:
            locations = np.where(np.isclose(morph_out[2], id+1))
            mask[locations] = 1

        ads.imwrite(Path(save_folder) / 'flagged_objjects.png', mask)
