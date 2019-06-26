# coding: utf-8

import pytest
import numpy as np
from imageio import imread
from AxonDeepSeg.data_management.data_augmentation import *


class TestCore(object):
    def setup(self):
        # Remember that the stop value in "arrange" is not included
        x = np.arange(0, 16, dtype='uint8')
        y = np.arange(0, 16, dtype='uint8')
        xv, yv = np.meshgrid(x, y)
        self.testImage = xv+yv

        self.mask = np.ones((16, 16, 3), dtype=int)

    def teardown(self):
        pass

    # --------------data_augmentation.py tests-------------- #
    # **NOTE** Because most data augmentation functions chose the parameters
    # randomly within bounds set by the arguments, it's impossible to target
    # test cases for known shifts, rotations, etc. Thus, only broad tests are
    # defined (output dim = input dim, output patch different or same as input
    # patch)

    @pytest.mark.unit
    def test_shifting_returns_different_image(self):
        patch = [self.testImage, self.mask]
        augmentedPatch = shifting(patch, verbose=1)

        assert augmentedPatch[0].shape == patch[0].shape
        assert augmentedPatch[1].shape == patch[1].shape
        assert np.not_equal(augmentedPatch[0], patch[0]).any()

    @pytest.mark.unit
    def test_rescaling_factor_1_returns_same_image(self):
        patch = [self.testImage, self.mask]
        augmentedPatch = rescaling(patch, factor_max=1, verbose=1)

        assert augmentedPatch[0].shape == patch[0].shape
        assert augmentedPatch[1].shape == patch[1].shape
        assert np.equal(augmentedPatch[0], patch[0]).all()

    @pytest.mark.unit
    def test_large_max_rescaling_factor_returns_different_image(self):
        # Note: Since there
        patch = [self.testImage, self.mask]
        augmentedPatch = rescaling(patch, factor_max=100, verbose=1)

        assert augmentedPatch[0].shape == patch[0].shape
        assert augmentedPatch[1].shape == patch[1].shape

        assert np.not_equal(augmentedPatch[0], patch[0]).any()
        # Note: due to  therandomized rescaling factor choice,
        # there is an unavoidable possibility that this test fails
        # simply due to bad luck, so try running the test suite again.

    @pytest.mark.unit
    def test_random_rotation_returns_different_image(self):
        patch = [self.testImage, self.mask]
        augmentedPatch = random_rotation(patch, verbose=1)

        assert augmentedPatch[0].shape == patch[0].shape
        assert augmentedPatch[1].shape == patch[1].shape
        assert np.not_equal(augmentedPatch[0], patch[0]).any()

    @pytest.mark.unit
    def test_elastic_returns_different_image(self):
        patch = [self.testImage, self.mask]
        augmentedPatch = elastic(patch, verbose=1)

        assert augmentedPatch[0].shape == patch[0].shape
        assert augmentedPatch[1].shape == patch[1].shape
        assert np.not_equal(augmentedPatch[0], patch[0]).any()

    @pytest.mark.unit
    def test_flipping_returns_different_image(self):
        # Since the flipping only occurs randomly thus has a probability of
        # not getting flipped, this low-level test only covers the same image
        # size assertions.

        patch = [self.testImage, self.mask]
        augmentedPatch = flipping(patch, verbose=1)

        assert augmentedPatch[0].shape == patch[0].shape
        assert augmentedPatch[1].shape == patch[1].shape

    @pytest.mark.unit
    def test_gaussian_blur_returns_image_of_same_size(self):
        # Because the sigma blur size is also random, it's very difficult to
        # test for a known case (different/same image), so this low-level test
        # only covers the same image size assertions.

        patch = [self.testImage, self.mask]
        augmentedPatch = gaussian_blur(patch, verbose=1)

        assert augmentedPatch[0].shape == patch[0].shape
        assert augmentedPatch[1].shape == patch[1].shape
