# coding: utf-8

import pytest
import os
import imageio
import numpy as np
import scipy as sp

from AxonDeepSeg.visualization.get_masks import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]

        self.path_folder = os.path.join(
            self.testPath,
            '__test_files__',
            '__test_demo_files__',
            '__prediction_only__'
            )

    def teardown(self):
        if os.path.isfile(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axon.png')):
            os.remove(
               os.path.join(self.path_folder, 'AxonDeepSeg_seg-axon.png')
               )

        if os.path.isfile(os.path.join(self.path_folder, 'AxonDeepSeg_seg-myelin.png')):
            os.remove(
               os.path.join(self.path_folder, 'AxonDeepSeg_seg-myelin.png')
               )

        if os.path.isfile(os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin-rgb.png')):
            os.remove(
               os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin-rgb.png')
               )

    # --------------get_masks tests-------------- #
    @pytest.mark.unit
    def test_get_masks_writes_expected_files(self):
        pred_img = os.path.join(
            self.path_folder,
            'AxonDeepSeg_seg-axonmyelin.png'
            )

        axon_prediction, myelin_prediction = get_masks(pred_img)

        axonFile = os.path.join(
            self.path_folder,
            'AxonDeepSeg_seg-axon.png'
            )

        myelinFile = os.path.join(
            self.path_folder,
            'AxonDeepSeg_seg-myelin.png'
            )

        assert os.path.isfile(axonFile)
        assert os.path.isfile(myelinFile)

    # --------------rgb_rendering_of_mask tests-------------- #
    @pytest.mark.unit
    def test_rgb_rendering_of_mask_returns_array_with_extra_dim_of_len_3(self):
        pred_img = imageio.imread(
            os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin.png')
            )

        rgb_mask = rgb_rendering_of_mask(pred_img)

        predShape = pred_img.shape
        rgbShape = rgb_mask.shape

        expectedRgbShape = predShape + (3,)

        assert rgbShape == expectedRgbShape

    @pytest.mark.unit
    def test_rgb_rendering_of_mask_writes_expected_files(self):
        pred_img = imageio.imread(
            os.path.join(self.path_folder, 'AxonDeepSeg_seg-axonmyelin.png')
            )

        rgbFile = os.path.join(
            self.path_folder,
            'AxonDeepSeg_seg-axonmyelin-rgb.png'
            )

        if os.path.isfile(rgbFile):
            os.remove(rgbFile)

        rgb_mask = rgb_rendering_of_mask(pred_img, rgbFile)

        assert os.path.isfile(rgbFile)

        assert np.array_equal(rgb_mask, imageio.imread(rgbFile))

    @pytest.mark.unit
    def test_get_image_properties_returns_expected_number_of_unique_values(self):
        pred_img = os.path.join(
            self.path_folder,
            'AxonDeepSeg_seg-axonmyelin.png'
            )
        
        image_properties = get_image_unique_vals_properties(pred_img)

        assert image_properties['num_uniques'] == 3
        assert np.array_equal(image_properties['unique_values'], [0, 127, 255])

    @pytest.mark.unit
    def test_get_image_properties_returns_expeception_for_unexpected_number_of_unique_values(self):
        pred_img = os.path.join(
            self.path_folder,
            'AxonDeepSeg_seg-axonmyelin.png'
            )

        loaded_image = imageio.imread(pred_img)

        image_properties = get_image_unique_vals_properties(loaded_image)

        assert image_properties['num_uniques'] == 3
        assert np.array_equal(image_properties['unique_values'], [0, 127, 255])


        # Resizing image with interpolation will add values to the image.
        resized_image = sp.misc.imresize(loaded_image, size=200 , interp='bilinear')

        resized_image_properties = get_image_unique_vals_properties(resized_image)

        assert not resized_image_properties['num_uniques'] == 3
