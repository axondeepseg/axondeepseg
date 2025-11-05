from pathlib import Path
import pytest
import tempfile
import shutil

import numpy as np

from AxonDeepSeg.visualization.imagej import roi_to_mask
from AxonDeepSeg import ads_utils as adsutils


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
        
        self.tempDir = Path(tempfile.mkdtemp())
        self.outputMaskPath = self.tempDir / 'test_mask.png'

    def teardown_method(self):
        if self.tempDir.exists():
            shutil.rmtree(self.tempDir)

    @pytest.mark.unit
    def test_roi_to_mask_creates_output_file(self):        

        roi_to_mask(self.testRoiFolder, self.testImagePath, self.outputMaskPath)
        
        assert self.outputMaskPath.exists(), "Output mask file was not created"
        
        mask = adsutils.imread(self.outputMaskPath)
        assert mask is not None, "Could not read the generated mask"
        assert isinstance(mask, np.ndarray), "Mask is not a numpy array"

    @pytest.mark.unit
    def test_roi_to_mask_binary_output(self):
        roi_to_mask(self.testRoiFolder, self.testImagePath, self.outputMaskPath)
        
        mask = adsutils.imread(self.outputMaskPath)
        unique_values = np.unique(mask)
        
        assert set(unique_values).issubset({0, 255}), f"Mask contains non-binary values: {unique_values}"

    @pytest.mark.unit
    def test_roi_to_mask_same_dimensions_as_input(self):
        if not self.testImagePath.exists():
            pytest.skip(f"Test image not found: {self.testImagePath}")
        if not self.testRoiFolder.exists():
            pytest.skip(f"Test ROI folder not found: {self.testRoiFolder}")
        
        input_image = adsutils.imread(self.testImagePath)
        input_shape = input_image.shape
        
        roi_to_mask(self.testRoiFolder, self.testImagePath, self.outputMaskPath)
        
        output_mask = adsutils.imread(self.outputMaskPath)
        output_shape = output_mask.shape
        
        assert input_shape == output_shape, (
            f"Mask shape {output_shape} doesn't match input image shape {input_shape}"
        )

    @pytest.mark.unit
    def test_roi_to_mask_empty_roi_folder(self):
        empty_folder = self.tempDir / 'empty_rois'
        empty_folder.mkdir()
        
        # This should run without error but create an empty mask
        roi_to_mask(empty_folder, self.testImagePath, self.outputMaskPath)
        
        assert self.outputMaskPath.exists(), "Output mask should be created even with empty ROI folder"
        
        # Check that mask is all zeros (empty)
        mask = adsutils.imread(self.outputMaskPath)
        assert np.all(mask == 0), "Mask should be all zeros with empty ROI folder"
