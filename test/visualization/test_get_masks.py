# coding: utf-8

from pathlib import Path
import imageio
import numpy as np
import os
from skimage.transform import resize
import pytest

from AxonDeepSeg.visualization.get_masks import *
from config import axonmyelin_suffix, axon_suffix, myelin_suffix


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
        if (self.path_folder / ('image' + str(axon_suffix))).is_file():
            (self.path_folder / ('image' + str(axon_suffix))).unlink()

        if (self.path_folder / ('image' + str(myelin_suffix))).is_file():
            (self.path_folder / ('image' + str(myelin_suffix))).unlink()

        if (self.path_folder / ('image' + axonmyelin_suffix.stem + '-rgb.png')).is_file():
            (self.path_folder / ('image' + axonmyelin_suffix.stem + '-rgb.png')).unlink()

    # --------------get_masks tests-------------- #
    @pytest.mark.unit
    def test_get_masks_writes_expected_files(self):
        pred_img = self.path_folder/ ('image' + str(axonmyelin_suffix))

        axon_prediction, myelin_prediction = get_masks(str(pred_img))

        axonFile = self.path_folder / ('image' + str(axon_suffix))

        myelinFile = self.path_folder / ('image' + str(myelin_suffix))

        assert axonFile.is_file()
        assert myelinFile.is_file()

    # --------------rgb_rendering_of_mask tests-------------- #
    @pytest.mark.unit
    def test_rgb_rendering_of_mask_returns_array_with_extra_dim_of_len_3(self):
        pred_img = imageio.imread(self.path_folder / ('image' + str(axonmyelin_suffix)))

        rgb_mask = rgb_rendering_of_mask(pred_img)

        predShape = pred_img.shape
        rgbShape = rgb_mask.shape
        expectedRgbShape = predShape + (3,)

        assert rgbShape == expectedRgbShape

    @pytest.mark.unit
    def test_rgb_rendering_of_mask_writes_expected_files(self):
        pred_img = imageio.imread(self.path_folder / ('image' + str(axonmyelin_suffix)))

        rgbFile = self.path_folder / ('image' + axonmyelin_suffix.stem + '-rgb.png')

        if rgbFile.is_file():
            rgbFile.unlink()

        rgb_mask = rgb_rendering_of_mask(pred_img, str(rgbFile))

        assert rgbFile.is_file()
        assert np.array_equal(rgb_mask, imageio.imread(rgbFile))

    # --------------get_image_properties tests-------------- #
    @pytest.mark.unit
    def test_get_image_properties_returns_expected_number_of_unique_values(self):
        pred_img = os.path.join(
            self.path_folder,
            ('image' + str(axonmyelin_suffix))
            )

        image_properties = get_image_unique_vals_properties(pred_img)

        assert image_properties['num_uniques'] == 3
        assert np.array_equal(image_properties['unique_values'], [0, 127, 255])

    @pytest.mark.unit
    def test_get_image_properties_returns_expeception_for_unexpected_number_of_unique_values(self):
        pred_img = os.path.join(
            self.path_folder,
            ('image' + str(axonmyelin_suffix))
            )

        loaded_image = imageio.imread(pred_img)

        image_properties = get_image_unique_vals_properties(loaded_image)

        assert image_properties['num_uniques'] == 3
        assert np.array_equal(image_properties['unique_values'], [0, 127, 255])


        # Resizing image with interpolation will add values to the image.
        resized_image = resize(loaded_image, (100, 100), order=1)

        resized_image_properties = get_image_unique_vals_properties(resized_image)

        assert not resized_image_properties['num_uniques'] == 3
