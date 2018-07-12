# coding: utf-8

import pytest
import os
import numpy as np
from scipy.misc import imread

from AxonDeepSeg.visualization.generate_axons_from_myelin import *

class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]

        self.folderPath = os.path.join(self.testPath, '__test_files__')
        self.axonMyelinMask = os.path.join(self.folderPath, 'AxonDeepSeg_seg-axonmyelin.png')
        self.myelinMask = os.path.join(self.folderPath, 'AxonDeepSeg_seg-myelin.png')

    def teardown(self):
        if os.path.isfile(os.path.join(self.folderPath,'axon_myelin_mask_corrected.png')):
            os.remove(os.path.join(self.folderPath,'axon_myelin_mask_corrected.png'))

    #--------------generate_axons_from_myelin tests--------------#
    @pytest.mark.current
    def test_generate_axons_from_myelin_creates_expected_file(self):

        if os.path.isfile(os.path.join(self.folderPath,'axon_myelin_mask_corrected.png')):
            os.remove(os.path.join(self.folderPath,'axon_myelin_mask_corrected.png'))

        generate_axons_from_myelin(self.axonMyelinMask, self.myelinMask)
        
        assert os.path.isfile(os.path.join(self.folderPath,'axon_myelin_mask_corrected.png'))
