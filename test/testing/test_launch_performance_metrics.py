# coding: utf-8

from pathlib import Path
import pytest

from ads_base.testing.launch_performance_metrics import launch_performance_metrics
from ads_base.params import axonmyelin_suffix


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.folderPath = self.testPath / '__test_files__'/'__test_demo_files__'

        self.prediction = self.folderPath/ ('image' + str(axonmyelin_suffix))

        self.groundtruth = self.folderPath / 'mask.png'

    def teardown_method(self):
        pass

    # --------------launch_performance_metrics tests-------------- #
    @pytest.mark.integration
    def test_launch_performance_metrics_runs_successfully(self):

        assert launch_performance_metrics(str(self.prediction), str(self.groundtruth))
