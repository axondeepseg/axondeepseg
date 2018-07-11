# coding: utf-8

import pytest
import os
import imageio
import numpy as np

from AxonDeepSeg.visualization.merge_masks import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]

        self.path_folder = os.path.join(self.testPath, '__test_files__')

    def teardown(self):
        if os.path.isfile(os.path.join(self.path_folder,'axon_myelin_mask.png')):
           os.remove(os.path.join(self.path_folder,'axon_myelin_mask.png'))

    #--------------merge_masks tests--------------#
    @pytest.mark.unit
    def test_merge_masks_outputs_expected_volume_and_writes_files(self):

        path_axon = os.path.join(self.path_folder,'AxonDeepSeg_seg-axon.png')
        path_myelin = os.path.join(self.path_folder,'AxonDeepSeg_seg-myelin.png')


        expectedFilePath = os.path.join(self.path_folder,'axon_myelin_mask.png')
        if os.path.isfile(expectedFilePath):
           os.remove(expectedFilePath)

        both = merge_masks(path_axon,path_myelin)

        assert os.path.isfile(expectedFilePath)

        assert np.array_equal(both, imageio.imread(expectedFilePath))
