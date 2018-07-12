# coding: utf-8

import pytest
import os
import numpy as np
from scipy.misc import imread

from AxonDeepSeg.testing.noise_simulation import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]

        self.folderPath = os.path.join(self.testPath, '__test_files__')

        self.image = imread(os.path.join(self.folderPath,'image.png'), flatten=True)

    def teardown(self):
        pass

    #--------------add_additive_gaussian_noise tests--------------#
    @pytest.mark.unittest
    def test_add_additive_gaussian_noise_returns_different_image_with_diff_std_near_sigma(self):

        sigma = 10

        noisyImage = add_additive_gaussian_noise(self.image, mu=0, sigma=sigma)

        assert not np.array_equal(noisyImage, self.image)
        assert abs(np.std(noisyImage-self.image) - sigma) < 1

    @pytest.mark.unittest
    def test_add_multiplicative_gaussian_noise_returns_different_image(self):

        sigma = 10

        noisyImage = add_multiplicative_gaussian_noise(self.image, mu=0, sigma=sigma)

        assert not np.array_equal(noisyImage, self.image)

    @pytest.mark.unittest
    def test_change_brightness(self):
        maxPixelValue = 255
        value_percentage=0.2
        
        brighterImage = change_brightness(self.image, value_percentage=value_percentage)

        assert not np.array_equal(brighterImage, self.image)

        assert abs(np.mean(brighterImage - self.image) - (value_percentage * maxPixelValue) ) < 0.1
