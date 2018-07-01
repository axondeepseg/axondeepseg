# coding: utf-8

import os
import shutil
import pytest
from AxonDeepSeg.data_management.dataset_building import *

class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))
        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]
        self.rawPath = os.path.join(self.testPath, '__test_patch_files__/raw')
        self.patchPath = os.path.join(self.testPath, '__test_patch_files__/patched')
        self.datasetPath = os.path.join(self.testPath, '__test_patch_files__/dataset')

    @classmethod
    def teardown_class(cls):
        fullPath = os.path.dirname(os.path.abspath(__file__))
        # Move up to the test directory, "test/"
        testPath = os.path.split(fullPath)[0]
        patchPath = os.path.join(testPath, '__test_patch_files__/patched')
        datasetPath = os.path.join(testPath, '__test_patch_files__/dataset')

        if os.path.exists(patchPath) and os.path.isdir(patchPath):
            shutil.rmtree(patchPath)

        if os.path.exists(datasetPath) and os.path.isdir(datasetPath):
            shutil.rmtree(datasetPath)

    #--------------dataset_building.py tests--------------#
    @pytest.mark.current
    def test_raw_img_to_patches_creates_expected_folders_and_files(self):
        if os.path.exists(self.patchPath) and os.path.isdir(self.patchPath):
            shutil.rmtree(self.patchPath)
        
        raw_img_to_patches(self.rawPath, self.patchPath)

        assert os.path.exists(self.patchPath) and os.path.isdir(self.patchPath)

        assert os.path.exists(self.patchPath+"/data1") and os.path.isdir(self.patchPath+"/data1")
        assert len(os.listdir(self.patchPath+"/data1")) == 12 # These demo image and mask are split into 6 patches each

        assert os.path.exists(self.patchPath+"/data2") and os.path.isdir(self.patchPath+"/data2")
        assert len(os.listdir(self.patchPath+"/data2")) == 24 # These demo image and mask are split into 12 patches each

    @pytest.mark.current
    def test_patched_to_dataset_creates_expected_folders_and_files(self):
        if os.path.exists(self.datasetPath) and os.path.isdir(self.datasetPath):
            shutil.rmtree(self.datasetPath)
        
        patched_to_dataset(self.patchPath, self.datasetPath, 'unique')

        assert os.path.exists(self.datasetPath) and os.path.isdir(self.datasetPath)
        assert len(os.listdir(self.datasetPath)) == 12+24 # Dataset folder merges all the patch folders generated in the test above.
