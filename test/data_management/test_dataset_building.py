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
        self.rawPath = os.path.join(self.testPath, '__test_files__/__test_patch_files__/raw')
        self.patchPath = os.path.join(self.testPath, '__test_files__/__test_patch_files__/patched')
        self.datasetPath = os.path.join(self.testPath, '__test_files__/__test_patch_files__/dataset')
        self.mixedPatchPath = os.path.join(self.testPath, '__test_files__/__test_patch_files__/mixedPatched')
        self.mixedDatasetPath = os.path.join(self.testPath, '__test_files__/__test_patch_files__/mixedDataset')

    @classmethod
    def teardown_class(cls):
        fullPath = os.path.dirname(os.path.abspath(__file__))
        # Move up to the test directory, "test/"
        testPath = os.path.split(fullPath)[0]
        patchPath = os.path.join(testPath, '__test_files__/__test_patch_files__/patched')
        datasetPath = os.path.join(testPath, '__test_files__/__test_patch_files__/dataset')
        
        mixedPatchPath = os.path.join(testPath, '__test_files__/__test_patch_files__/mixedPatched')
        mixedDatasetPath = os.path.join(testPath, '__test_files__/__test_patch_files__/mixedDataset')

        if os.path.exists(patchPath) and os.path.isdir(patchPath):
            shutil.rmtree(patchPath)

        if os.path.exists(datasetPath) and os.path.isdir(datasetPath):
            shutil.rmtree(datasetPath)

        if os.path.exists(mixedPatchPath) and os.path.isdir(mixedPatchPath):
            shutil.rmtree(mixedPatchPath)

        if os.path.exists(mixedDatasetPath) and os.path.isdir(mixedDatasetPath):
            shutil.rmtree(mixedDatasetPath)

    #--------------raw_img_to_patches tests--------------#
    @pytest.mark.unit
    def test_raw_img_to_patches_creates_expected_folders_and_files(self):
        if os.path.exists(self.patchPath) and os.path.isdir(self.patchPath):
            shutil.rmtree(self.patchPath)
        
        raw_img_to_patches(self.rawPath, self.patchPath)

        assert os.path.exists(self.patchPath) and os.path.isdir(self.patchPath)

        assert os.path.exists(self.patchPath+"/data1") and os.path.isdir(self.patchPath+"/data1")
        assert len(os.listdir(self.patchPath+"/data1")) == 12 # These demo image and mask are split into 6 patches each

        assert os.path.exists(self.patchPath+"/data2") and os.path.isdir(self.patchPath+"/data2")
        assert len(os.listdir(self.patchPath+"/data2")) == 24 # These demo image and mask are split into 12 patches each

    #--------------patched_to_dataset tests--------------#
    @pytest.mark.unit
    def test_patched_to_dataset_creates_expected_folders_and_files(self):
        if os.path.exists(self.datasetPath) and os.path.isdir(self.datasetPath):
            shutil.rmtree(self.datasetPath)
        
        patched_to_dataset(self.patchPath, self.datasetPath, 'unique')

        assert os.path.exists(self.datasetPath) and os.path.isdir(self.datasetPath)
        assert len(os.listdir(self.datasetPath)) == 12+24 # Dataset folder merges all the patch folders generated in the test above.

    @pytest.mark.unit
    def test_patched_to_dataset_fake_mixed_dataset_creates_expected_folders_and_files(self):
        # TEM images are too large to be included in repo (6+ megs), so simply create fake duplicate dataset with SEM images.
        if os.path.exists(self.mixedDatasetPath) and os.path.isdir(self.mixedDatasetPath):
            shutil.rmtree(self.mixedDatasetPath)
        
        raw_img_to_patches(self.rawPath, os.path.join(self.mixedPatchPath,'SEM'))
        raw_img_to_patches(self.rawPath, os.path.join(self.mixedPatchPath,'TEM'))

        patched_to_dataset(self.mixedPatchPath, self.mixedDatasetPath, 'mixed')

        assert os.path.exists(self.mixedDatasetPath) and os.path.isdir(self.mixedDatasetPath)
        assert len(os.listdir(self.mixedDatasetPath)) == (12+24)*2 # Dataset folder merges all the patch folders generated in the test above.
