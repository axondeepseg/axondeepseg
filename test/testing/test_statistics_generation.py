# coding: utf-8

import pytest
import json
import os
import tensorflow as tf
import pandas as pd
from AxonDeepSeg.testing.statistics_generation import *


class TestCore(object):
    def setup(self):

        self.fullPath = os.path.dirname(os.path.abspath(__file__))
        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]
        self.projectPath = os.path.split(self.testPath)[0]

        self.modelPath = os.path.join(
            self.projectPath,
            'AxonDeepSeg',
            'models',
            'default_SEM_model_v1'
            )

        self.imagesPath = os.path.join(
            self.testPath,
            '__test_files__',
            '__test_training_files__',
            'Testing'
            )

        self.statsFilename = 'model_statistics_validation.json'

    @classmethod
    def teardown_class(cls):
        fullPath = os.path.dirname(os.path.abspath(__file__))
        # Move up to the test directory, "test/"
        testPath = os.path.split(fullPath)[0]
        projectPath = os.path.split(testPath)[0]

        modelPath = os.path.join(
            projectPath,
            'AxonDeepSeg',
            'models',
            'default_SEM_model_v1'
            )

        statsFilename = 'model_statistics_validation.json'

        if os.path.exists(os.path.join(modelPath, statsFilename)):
            os.remove(os.path.join(modelPath, statsFilename))

    # --------------metrics_single_wrapper tests-------------- #
    @pytest.mark.integration
    def test_metrics_single_wrapper_runs_succesfully_and_outfile_exists(self):
        # reset the tensorflow graph for new training
        tf.reset_default_graph()

        path_model_folder = self.modelPath
        path_images_folder = self.imagesPath
        resampled_resolution = 0.1
        metrics_single_wrapper(
            path_model_folder,
            path_images_folder,
            resampled_resolution,
            overlap_value=25,
            statistics_filename=self.statsFilename,
            create_statistics_file=True,
            verbosity_level=2
            )

        assert os.path.exists(os.path.join(self.modelPath, self.statsFilename))

    # --------------metrics_classic_wrapper tests-------------- #
    @pytest.mark.integration
    def test_metrics_classic_wrapper_runs_succesfully_and_outfile_exists(self):
        # reset the tensorflow graph for new training
        tf.reset_default_graph()

        path_model_folder = self.modelPath
        path_images_folder = self.imagesPath
        resampled_resolution = 0.1
        metrics_classic_wrapper(
            path_model_folder,
            path_images_folder,
            resampled_resolution,
            overlap_value=25,
            statistics_filename=self.statsFilename,
            create_statistics_file=True,
            verbosity_level=2)

        assert os.path.exists(os.path.join(self.modelPath, self.statsFilename))

    # --------------metrics class tests-------------- #
    # Though conceptually these could be classified as unit tests, they
    # depend on the outputs of the previous integrity tests, so we count these
    # as part of the same integrity test workflow
    @pytest.mark.integration
    def test_metrics_class_loads_stats_table(self):
        met = metrics(statistics_filename='model_statistics_validation.json')

        assert type(met.filtered_stats) is pd.core.frame.DataFrame
        assert met.filtered_stats.empty

        met.add_models(self.modelPath)
        met.load_models()

        assert not met.filtered_stats.empty

    @pytest.mark.integration
    def test_metrics_class_throws_exception_for_missing_stats_file(self):
        met = metrics(statistics_filename='n0n-3x1st1ng.json')

        met.add_models(self.modelPath)

        with pytest.raises(ValueError):
            met.load_models()
