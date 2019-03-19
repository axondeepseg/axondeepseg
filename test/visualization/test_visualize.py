# coding: utf-8

from pathilb import Path
import numpy as np
from scipy.misc import imread

import pytest

from AxonDeepSeg.visualization.visualize import *


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

        assert visualize_training(self.pathModel)
