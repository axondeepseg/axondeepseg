# coding: utf-8

import pytest
import numpy as np
from scipy.misc import imread
from AxonDeepSeg.data_management.patch_extraction import extract_patch

class TestCore(object):
    def setup(self):
        x=np.arange(0,16,dtype='uint8') # The stop value in "arrange" is not included
        y=np.arange(0,16,dtype='uint8')
        xv, yv = np.meshgrid(x, y)
        self.testImage = xv+yv

        self.mask = np.ones((16,16,3),dtype=int)

    def teardown(self):
        pass

    #--------------patch_extraction.py tests--------------#
    @pytest.mark.current
    def test_extract_patch_script_returns_expected_patches(self):
        to_extract = [self.testImage, self.mask]
        patch_size = 4
        print self.testImage
        output = extract_patch(to_extract, patch_size)
        
        # for debugging only
        for j, patch in enumerate(output):
            print j
            print patch[0]

        expectedTopLeftPatch = self.testImage[0:4,0:4]
        expectedTopRightPatch = self.testImage[0:4,12:16]

        expectedSecondRowFirstPatch = self.testImage[4:8,0:4]

        print output[0][0]
        print expectedTopLeftPatch
        assert np.array_equal(output[0][0], expectedTopLeftPatch)
        assert np.array_equal(output[4][0], expectedTopRightPatch)
        assert np.array_equal(output[5][0], expectedSecondRowFirstPatch)
