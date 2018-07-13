# coding: utf-8

import pytest
import os
import imageio
import numpy as np

from AxonDeepSeg.visualization.get_masks import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]

        self.path_folder = os.path.join(self.testPath, '__test_files__/__test_demo_files__/__prediction_only__')

    def teardown(self):
        if os.path.isfile(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axon.png')):
           os.remove(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axon.png'))

        if os.path.isfile(os.path.join(self.path_folder, 'AxonDeepSeg_seg-myelin.png')):
           os.remove(os.path.join(self.path_folder, 'AxonDeepSeg_seg-myelin.png'))

        if os.path.isfile(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin-rgb.png')):
           os.remove(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin-rgb.png'))

    #--------------get_mask tests--------------#
    @pytest.mark.unittest
    def test_get_masks_writes_expected_files(self):
        pred_img=os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin.png')

        axon_prediction, myelin_prediction = get_masks(pred_img)

        axonFile = os.path.join(self.path_folder, 'AxonDeepSeg_seg-axon.png')
        myelinFile = os.path.join(self.path_folder, 'AxonDeepSeg_seg-myelin.png')

        assert os.path.isfile(axonFile)
        assert os.path.isfile(myelinFile)


    #--------------rgb_rendering_of_mask tests--------------#
    @pytest.mark.unittest
    def test_rgb_rendering_of_mask_returns_array_with_extra_dim_of_len_3(self):
        pred_img=imageio.imread(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin.png'))

        rgb_mask = rgb_rendering_of_mask(pred_img)

        predShape = pred_img.shape
        rgbShape = rgb_mask.shape

        expectedRgbShape = predShape + (3,)

        assert rgbShape == expectedRgbShape

    @pytest.mark.unittest
    def test_rgb_rendering_of_mask_writes_expected_files(self):
        pred_img=imageio.imread(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin.png'))

        rgbFile = os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin-rgb.png')

        if os.path.isfile(rgbFile):
           os.remove(rgbFile)

        rgb_mask = rgb_rendering_of_mask(pred_img, rgbFile)

        assert os.path.isfile(rgbFile)
        
        assert np.array_equal(rgb_mask, imageio.imread(rgbFile))
