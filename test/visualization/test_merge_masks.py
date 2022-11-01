# coding: utf-8

from pathlib import Path
import imageio
import numpy as np

import pytest

from AxonDeepSeg.visualization.merge_masks import merge_masks
from config import axon_suffix, myelin_suffix, axonmyelin_suffix


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent
        self.output_filename =  Path('image' + str(axonmyelin_suffix))
        self.path_folder = (
            self.testPath /
            '__test_files__' /
            '__test_demo_files__'
            )

    def teardown_method(self):
        if (self.path_folder / self.output_filename ).is_file():
            (self.path_folder / self.output_filename ).unlink()

    # --------------merge_masks tests-------------- #
    @pytest.mark.unit
    def test_merge_masks_outputs_expected_volume_and_writes_files(self):

        path_axon = self.path_folder / ('image' + str(axon_suffix))

        path_myelin = self.path_folder / ('image' + str(myelin_suffix))

        expectedFilePath = self.path_folder / self.output_filename 

        if expectedFilePath.is_file():
            expectedFilePath.unlink()

        both = merge_masks(str(path_axon), str(path_myelin), str(self.output_filename))

        assert expectedFilePath.is_file()
        assert np.array_equal(both, imageio.imread(expectedFilePath))
