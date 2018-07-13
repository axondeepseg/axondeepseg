# coding: utf-8

import pytest
import numpy as np
from scipy.misc import imread
from AxonDeepSeg.data_management.patch_extraction import extract_patch


class TestCore(object):
    def setup(self):
        # Remember: the stop value in "arrange" is not included
        x = np.arange(0, 16, dtype='uint8')
        y = np.arange(0, 16, dtype='uint8')
        xv, yv = np.meshgrid(x, y)
        self.testImage = xv+yv

        self.mask = np.ones((16, 16, 3), dtype=int)

    def teardown(self):
        pass

    # --------------extract_patch tests-------------- #
    @pytest.mark.unit
    def test_extract_patch_script_returns_expected_patches(self):
        to_extract = [self.testImage, self.mask]
        patch_size = 4

        output = extract_patch(to_extract, patch_size)

        expectedTopLeftPatch = self.testImage[0:4, 0:4]
        expectedBottomRightPatch = self.testImage[12:16, 12:16]

        assert np.array_equal(output[0][0], expectedTopLeftPatch)
        assert np.array_equal(output[-1][0], expectedBottomRightPatch)

        # Current implementation of patch extration in ADS contains some
        # overlap which is not specified/controllable, so other cases are
        # difficult to test. When a controllable overlap is implemented, add
        # more tests accordingly.

    @pytest.mark.unit
    def test_extract_patch_script_errors_for_patch_size_smaller_than_3(self):
        to_extract = [self.testImage, self.mask]
        patch_size = 2

        with pytest.raises(ValueError):
            extract_patch(to_extract, patch_size)

    @pytest.mark.unit
    def test_extract_patch_script_errors_for_patch_size_eq_to_image_dim(self):
        to_extract = [self.testImage, self.mask]
        patch_size = min(self.testImage.shape)

        with pytest.raises(ValueError):
            extract_patch(to_extract, patch_size)

    @pytest.mark.unit
    def test_extract_patch_script_errors_for_incorrect_first_arg_format(self):
        to_extract = self.testImage
        patch_size = 4

        with pytest.raises(ValueError):
            extract_patch(to_extract, patch_size)
