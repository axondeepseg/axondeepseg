# coding: utf-8

from pathlib import Path

import pytest

from AxonDeepSeg.visualization.visualize import visualize_training


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.pathModel = (
            self.testPath /
            '__test_files__' /
            '__test_model__' /
            'Model'
            )

    def teardown(self):
        pass

    # --------------visualize_training tests-------------- #
    @pytest.mark.unit
    def test_visualize_training_runs_successfully(self):

        assert visualize_training(str(self.pathModel))
