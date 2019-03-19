# coding: utf-8

from pathlib import Path
import imageio
import numpy as np

import pytest

from AxonDeepSeg.visualization.get_masks import *


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.path_folder = (
            self.testPath /
            '__test_files__' /
            '__test_demo_files__' /
            '__prediction_only__'
            )

    def teardown(self):
        if (self.path_folder / 'AxonDeepSeg_seg-axon.png').is_file():
            (self.path_folder / 'AxonDeepSeg_seg-axon.png').unlink()

        if (self.path_folder / 'AxonDeepSeg_seg-myelin.png').is_file():
            (self.path_folder / 'AxonDeepSeg_seg-myelin.png').unlink()

        if (self.path_folder / 'AxonDeepSeg_seg-axonmyelin-rgb.png').is_file():
            (self.path_folder / 'AxonDeepSeg_seg-axonmyelin-rgb.png').unlink()

    # --------------get_masks tests-------------- #
    @pytest.mark.unit
    def test_get_masks_writes_expected_files(self):
        pred_img = self.path_folder/ 'AxonDeepSeg_seg-axonmyelin.png'

        axon_prediction, myelin_prediction = get_masks(pred_img)

        axonFile = self.path_folder / 'AxonDeepSeg_seg-axon.png'

        myelinFile = self.path_folder / 'AxonDeepSeg_seg-myelin.png'

        assert axonFile.is_file()
        assert myelinFile.is_file()

    # --------------rgb_rendering_of_mask tests-------------- #
    @pytest.mark.unit
    def test_rgb_rendering_of_mask_returns_array_with_extra_dim_of_len_3(self):
        pred_img = imageio.imread(self.path_folder / 'AxonDeepSeg_seg-axonmyelin.png')

        rgb_mask = rgb_rendering_of_mask(pred_img)

        predShape = pred_img.shape
        rgbShape = rgb_mask.shape
        expectedRgbShape = predShape + (3,)

        assert rgbShape == expectedRgbShape

    @pytest.mark.unit
    def test_rgb_rendering_of_mask_writes_expected_files(self):
        pred_img = imageio.imread(self.path_folder / 'AxonDeepSeg_seg-axonmyelin.png')

        rgbFile = self.path_folder / 'AxonDeepSeg_seg-axonmyelin-rgb.png'

        if rgbFile.is_file():
            rgbFile.unlink()

        rgb_mask = rgb_rendering_of_mask(pred_img, rgbFile)

        assert rgbFile.is_file()
        assert np.array_equal(rgb_mask, imageio.imread(rgbFile))
