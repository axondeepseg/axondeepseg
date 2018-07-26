# coding: utf-8

import pytest
import os
import random
import string

from AxonDeepSeg.morphometrics.launch_morphometrics_computation import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))
        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]
        self.dataPath = os.path.join(
            self.testPath,
            '__test_files__',
            '__test_demo_files__'
            )

    def teardown(self):
        pass

    # --------------launch_morphometrics_computation tests-------------- #
    @pytest.mark.unit
    def test_launch_morphometrics_computation_saves_expected_files(self):
        expectedFiles = {'aggregate_morphometrics.txt',
                         'AxonDeepSeg_map-axondiameter.png',
                         'axonlist.npy'
                         }

        pathImg = os.path.join(self.dataPath, 'image.png')
        pathPrediction = os.path.join(self.dataPath,
                                      'AxonDeepSeg_seg-axonmyelin.png')

        launch_morphometrics_computation(pathImg, pathPrediction)

        for fileName in expectedFiles:
            fullFilePath = os.path.join(self.dataPath, fileName)
            assert os.path.isfile(fullFilePath)
            os.remove(fullFilePath)

    @pytest.mark.unit
    def test_launch_morphometrics_computation_errors_for_missing_file(self):
        nonExistingFile = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        pathPrediction = os.path.join(
            self.dataPath,
            'AxonDeepSeg_seg-axonmyelin.png'
            )

        with pytest.raises((IOError, OSError)):
            launch_morphometrics_computation(nonExistingFile, pathPrediction)
