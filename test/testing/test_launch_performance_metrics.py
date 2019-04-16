# coding: utf-8

from pathlib import Path
import pytest

from AxonDeepSeg.testing.launch_performance_metrics import *


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.folderPath = self.testPath / '__test_files__'/'__test_demo_files__'

        self.prediction = self.folderPath/ 'AxonDeepSeg_seg-axonmyelin.png'

        self.groundtruth = self.folderPath / 'mask.png'

    def teardown(self):
        pass

    # --------------launch_performance_metrics tests-------------- #
    @pytest.mark.integration
    def test_launch_performance_metrics_runs_successfully(self):

        assert launch_performance_metrics(str(self.prediction), str(self.groundtruth))
