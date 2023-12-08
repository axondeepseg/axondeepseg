# coding: utf-8

from pathlib import Path
import numpy as np
import imageio
import pytest

from AxonDeepSeg import ads_utils
from AxonDeepSeg.testing.noise_simulation import add_additive_gaussian_noise, add_multiplicative_gaussian_noise, change_brightness


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.folderPath = self.testPath / '__test_files__' / '__test_demo_files__'

        self.image = ads_utils.imread(self.folderPath / 'image.png')

    def teardown_method(self):
        pass

    # --------------add_additive_gaussian_noise tests-------------- #
    @pytest.mark.unit
    def test_add_additive_gaussian_noise_returns_expected_std_diff(self):

        sigma = 10

        noisyImage = add_additive_gaussian_noise(self.image, mu=0, sigma=sigma)

        assert not np.array_equal(noisyImage, self.image)
        assert abs(np.std(noisyImage-self.image) - sigma) < 1

    # --------------add_multiplicative_gaussian_noise tests-------------- #
    @pytest.mark.unit
    def test_add_multiplicative_gaussian_noise_returns_different_image(self):

        sigma = 10

        noisyImage = add_multiplicative_gaussian_noise(
            self.image,
            mu=0,
            sigma=sigma
            )

        assert not np.array_equal(noisyImage, self.image)

    # --------------change_brightness tests-------------- #
    @pytest.mark.unit
    def test_change_brightness(self):
        maxPixelValue = 255
        value_percentage=0.2

        brighterImage = change_brightness(
            self.image,
            value_percentage=value_percentage
            )

        assert not np.array_equal(brighterImage, self.image)

        assert abs(np.mean(brighterImage - self.image) - (value_percentage * maxPixelValue)) < 0.1
