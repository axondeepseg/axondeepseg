# coding: utf-8

from pathilb import Path
import imageio
import numpy as np

import pytest

from AxonDeepSeg.visualization.merge_masks import *


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.path_folder = (
            self.testPath /
            '__test_files__' /
            '__test_demo_files__'
            )

    def teardown(self):
        if (self.path_folder / 'axon_myelin_mask.png').is_file():
            (self.path_folder / 'axon_myelin_mask.png').unlink()

    # --------------merge_masks tests-------------- #
    @pytest.mark.unit
    def test_merge_masks_outputs_expected_volume_and_writes_files(self):

        path_axon = self.path_folder / 'AxonDeepSeg_seg-axon.png'

        path_myelin = self.path_folder / 'AxonDeepSeg_seg-myelin.png'

        expectedFilePath = self.path_folder / 'axon_myelin_mask.png'

        if expectedFilePath.is_file():
            expectedFilePath.unlink()

        both = merge_masks(path_axon, path_myelin)

        assert expectedFilePath.is_file()
        assert np.array_equal(both, imageio.imread(expectedFilePath))
