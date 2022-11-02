# coding: utf-8

from pathlib import Path
import numpy as np
from imageio import imread

import pytest

from AxonDeepSeg.visualization.generate_axons_from_myelin import generate_axons_from_myelin
from config import myelin_suffix, axonmyelin_suffix

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.folderPath = self.testPath / '__test_files__'/ '__test_demo_files__'

        self.axonMyelinMask = self.folderPath / ('image' + str(axonmyelin_suffix))

        self.myelinMask = self.folderPath / ('image' + str(myelin_suffix))

    def teardown_method(self):
        if (self.folderPath / 'axon_myelin_mask_corrected.png').is_file():
            (self.folderPath / 'axon_myelin_mask_corrected.png').unlink()

    # --------------generate_axons_from_myelin tests-------------- #
    @pytest.mark.unit
    def test_generate_axons_from_myelin_creates_expected_file(self):

        if (self.folderPath / 'axon_myelin_mask_corrected.png').is_file():
                (self.folderPath / 'axon_myelin_mask_corrected.png').unlink()

        generate_axons_from_myelin(str(self.axonMyelinMask), str(self.myelinMask))

        assert (self.folderPath / 'axon_myelin_mask_corrected.png').is_file()
