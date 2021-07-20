# coding: utf-8

import json
from pathlib import Path
import pandas as pd

import pytest

from AxonDeepSeg.testing.statistics_generation import metrics_single_wrapper, metrics, metrics_classic_wrapper


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent
        self.projectPath = self.testPath.parent

        self.modelPath = (
            self.projectPath /
            'AxonDeepSeg' /
            'models' /
            'default_SEM_model'
            )

        self.imagesPath = (
            self.testPath /
            '__test_files__' /
            '__test_training_files__' /
            'Testing'
            )

        self.statsFilename = 'model_statistics_validation.json'

    @classmethod
    def teardown_class(cls):
         # Get the directory where this current file is saved
        fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        testPath = fullPath.parent
        projectPath = testPath.parent

        modelPath = (
            projectPath /
            'AxonDeepSeg' /
            'models' /
            'default_SEM_model'
            )

        statsFilename = 'model_statistics_validation.json'

        if (modelPath / statsFilename).exists():
            (modelPath / statsFilename).unlink()

    # --------------metrics_single_wrapper tests-------------- #
    @pytest.mark.integration
    def test_metrics_single_wrapper_runs_successfully_and_outfile_exists(self):
        # reset the tensorflow graph for new training
        #tf.reset_default_graph()

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

        assert (self.modelPath / self.statsFilename).exists()
        (self.modelPath / self.statsFilename).unlink()
    
    # --------------metrics_classic_wrapper tests-------------- #
    @pytest.mark.integration
    def test_metrics_classic_wrapper_runs_successfully_and_outfile_exists(self):
        # reset the tensorflow graph for new training
        #tf.reset_default_graph()

        path_model_folder = self.modelPath
        path_images_folder = self.imagesPath
        resampled_resolution = 0.1
        metrics_classic_wrapper(
            str(path_model_folder),
            str(path_images_folder),
            resampled_resolution,
            overlap_value=25,
            statistics_filename=self.statsFilename,
            create_statistics_file=True,
            verbosity_level=2)

        assert (self.modelPath /self.statsFilename).exists()

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
