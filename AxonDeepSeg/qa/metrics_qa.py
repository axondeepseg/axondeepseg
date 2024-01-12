# coding: utf-8

# Scientific modules imports
import numpy as np

# Graphs and plots imports
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

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

    def plot(self, metric_name):
        x = self.df[metric_name].to_numpy()

        # bins='fd' uses the Freedman Diaconis Estimator to find the optimal number of bins
        count, bins, ignored = plt.hist(x,bins='fd', histtype='bar', ec='black')

        plt.show()
    
