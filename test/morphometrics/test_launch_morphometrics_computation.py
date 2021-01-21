# coding: utf-8

from pathlib import Path
import random
import string
import pytest

from AxonDeepSeg.morphometrics.launch_morphometrics_computation import *


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent
        self.dataPath = self.testPath / '__test_files__' / '__test_demo_files__'

    def teardown(self):
        pass

    # --------------launch_morphometrics_computation tests-------------- #
    @pytest.mark.unit
    def test_launch_morphometrics_computation_saves_expected_files(self):
        expectedFiles = {'aggregate_morphometrics.txt',
                         'AxonDeepSeg_map-axondiameter.png',
                         'axonlist.npy'
                         }

        pathImg = self.dataPath / 'image.png'
        pathPrediction = self.dataPath / 'image_seg-axonmyelin.png'

        launch_morphometrics_computation(str(pathImg), str(pathPrediction))

        for fileName in expectedFiles:
            print(fileName)
            fullFilePath = self.dataPath / fileName
            assert fullFilePath.is_file()
            fullFilePath.unlink()

    @pytest.mark.unit
    def test_launch_morphometrics_computation_errors_for_missing_file(self):
        nonExistingFile = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        pathPrediction = self.dataPath / 'image_seg-axonmyelin.png'

        with pytest.raises((IOError, OSError)):
            launch_morphometrics_computation(str(nonExistingFile), str(pathPrediction))
