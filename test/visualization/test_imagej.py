from pathlib import Path
import pytest
import tempfile
import shutil

import numpy as np

from AxonDeepSeg.visualization.imagej import roi_to_masks
from AxonDeepSeg import ads_utils as adsutils
from AxonDeepSeg.params import axonmyelin_suffix, axon_suffix, myelin_suffix

class TestImageJ(object):
    def setup_method(self):
        self.fullPath = Path(__file__).resolve().parent
        self.testPath = self.fullPath.parent
        
        self.testFilesPath = (
            self.testPath / 
            '__test_files__' / 
            '__test_demo_files__'
        )
        
        self.testImagePath = self.testFilesPath / 'image.png'
        self.testRoiFolder = self.testFilesPath / 'rois'
        
        self.axonmyelinMaskPath = self.testFilesPath / f'image{axonmyelin_suffix}'
        self.axonMaskPath = self.testFilesPath / f'image{axon_suffix}'
        self.myelinMaskPath = self.testFilesPath / f'image{myelin_suffix}'

    def teardown_method(self):
        if  self.axonmyelinMaskPath.exists():
            self.axonmyelinMaskPath.unlink()

        if  self.axonMaskPath.exists():
            self.axonMaskPath.unlink()

        if  self.myelinMaskPath.exists():
            self.myelinMaskPath.unlink()

    @pytest.mark.unit
    def test_roi_to_masks_creates_output_file(self):        

        roi_to_masks(self.testRoiFolder, self.testImagePath)
        
        assert self.axonmyelinMaskPath.exists(), "AxonMyelin mask file was not created"
        assert self.axonMaskPath.exists(), "Axon mask file was not created"
        assert self.myelinMaskPath.exists(), "Myelin mask file was not created"

        self.axonmyelinMaskPath.unlink()
        assert not self.axonmyelinMaskPath.exists(), "AxonMyelin mask file was not deleted"

        self.axonMaskPath.unlink()
        assert not self.axonMaskPath.exists(), "Axon mask file was not deleted"

        self.myelinMaskPath.unlink()
        assert not self.myelinMaskPath.exists(), "Myelin mask file was not deleted"


    @pytest.mark.unit
    def test_roi_to_masks_binary_output(self):
        roi_to_masks(self.testRoiFolder, self.testImagePath)
        
        mask = adsutils.imread(self.axonmyelinMaskPath)
        unique_values = np.unique(mask)
        
        assert set(unique_values).issubset({0, 255}), f"Mask contains non-binary values: {unique_values}"

        self.axonmyelinMaskPath.unlink()
        assert not self.axonmyelinMaskPath.exists(), "AxonMyelin mask file was not deleted"

        self.axonMaskPath.unlink()
        assert not self.axonMaskPath.exists(), "Axon mask file was not deleted"

        self.myelinMaskPath.unlink()
        assert not self.myelinMaskPath.exists(), "Myelin mask file was not deleted"


    @pytest.mark.unit
    def test_roi_to_masks_same_dimensions_as_input(self):
        if not self.testImagePath.exists():
            pytest.skip(f"Test image not found: {self.testImagePath}")
        if not self.testRoiFolder.exists():
            pytest.skip(f"Test ROI folder not found: {self.testRoiFolder}")
        
        input_image = adsutils.imread(self.testImagePath)
        input_shape = input_image.shape
        
        roi_to_masks(self.testRoiFolder, self.testImagePath)
        
        output_mask = adsutils.imread(self.axonmyelinMaskPath)
        output_shape = output_mask.shape
        
        assert input_shape == output_shape, (
            f"Mask shape {output_shape} doesn't match input image shape {input_shape}"
        )

        self.axonmyelinMaskPath.unlink()
        assert not self.axonmyelinMaskPath.exists(), "AxonMyelin mask file was not deleted"

        self.axonMaskPath.unlink()
        assert not self.axonMaskPath.exists(), "Axon mask file was not deleted"

        self.myelinMaskPath.unlink()
        assert not self.myelinMaskPath.exists(), "Myelin mask file was not deleted"
