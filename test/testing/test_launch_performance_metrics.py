# coding: utf-8

import pytest
import os
from AxonDeepSeg.testing.launch_performance_metrics import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]

        self.folderPath = os.path.join(self.testPath, '__test_files__/__test_demo_files__')

        self.prediction = os.path.join(self.folderPath,'AxonDeepSeg_seg-axonmyelin.png')
        self.groundtruth = os.path.join(self.folderPath,'mask.png')

    def teardown(self):
        pass

    #--------------launch_performance_metrics tests--------------#
    @pytest.mark.integritytest
    def test_launch_performance_metrics_runs_succesfully(self):

        assert launch_performance_metrics(self.prediction, self.groundtruth)
