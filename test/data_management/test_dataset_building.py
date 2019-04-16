# coding: utf-8

from pathlib import Path
import shutil
import pytest
from AxonDeepSeg.data_management.dataset_building import *


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        _create_new_test_folder = (
            lambda s: self.testPath
            / '__test_files__'
            / '__test_patch_files__'
            / s
            )

        self.rawPath =  _create_new_test_folder('raw')
        self.patchPath =  _create_new_test_folder('patched')
        self.datasetPath = _create_new_test_folder('dataset')
        self.mixedPatchPath = _create_new_test_folder('mixedPatched')
        self.mixedDatasetPath = _create_new_test_folder('mixedDataset')

    @classmethod
    def teardown_class(cls):
        # Get the directory where this current file is saved
        fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        testPath = fullPath.parent

        _create_new_test_folder = (
            lambda s: testPath
            / '__test_files__'
            / '__test_patch_files__'
            / s
            )

        patchPath =  _create_new_test_folder('patched')
        datasetPath = _create_new_test_folder('dataset')
        mixedPatchPath = _create_new_test_folder('mixedPatched')
        mixedDatasetPath = _create_new_test_folder('mixedDataset')

        if patchPath.is_dir():
            shutil.rmtree(patchPath)

        if datasetPath.is_dir():
            shutil.rmtree(datasetPath)

        if mixedPatchPath.is_dir():
            shutil.rmtree(mixedPatchPath)

        if mixedDatasetPath.is_dir():
            shutil.rmtree(mixedDatasetPath)

    # --------------raw_img_to_patches tests-------------- #
    @pytest.mark.unit
    def test_raw_img_to_patches_creates_expected_folders_and_files(self):
        if self.patchPath.is_dir():
            shutil.rmtree(self.patchPath)

        raw_img_to_patches(str(self.rawPath), str(self.patchPath))

        assert self.patchPath.is_dir()

        # These demo image and mask are split into 6 patches each
        path_to_data1 = self.patchPath / 'data1'
        assert(path_to_data1.is_dir())
        assert len([item for item in path_to_data1.iterdir()]) == 12

        # These demo image and mask are split into 12 patches each
        path_to_data2 = self.patchPath / 'data2'
        assert(path_to_data2.is_dir())
        assert len([item for item in path_to_data2.iterdir()]) == 24

    # --------------patched_to_dataset tests-------------- #
    @pytest.mark.unit
    def test_patched_to_dataset_creates_expected_folders_and_files(self):
        if self.datasetPath.is_dir():
            shutil.rmtree(self.datasetPath)

        patched_to_dataset(str(self.patchPath), str(self.datasetPath), 'unique')

        assert self.datasetPath.is_dir()

        # Dataset folder merges all the patch folders generated
        assert len([item for item in self.datasetPath.iterdir()]) == 12+24

    @pytest.mark.unit
    def test_patched_to_dataset_fake_mixed_dataset_creates_expected_dir(self):
        # TEM images are too large to be included in repo (6+ megs), so simply
        # create fake duplicate dataset with SEM images.
        if self.mixedDatasetPath.is_dir():
            shutil.rmtree(self.mixedDatasetPath)

        raw_img_to_patches(str(self.rawPath), str(self.mixedPatchPath / 'SEM'))

        raw_img_to_patches(str(self.rawPath), str(self.mixedPatchPath / 'TEM'))

        patched_to_dataset(str(self.mixedPatchPath), str(self.mixedDatasetPath), 'mixed')

        assert self.mixedDatasetPath.is_dir()

        # Dataset folder merges all the patch folders generated
        assert len([item for item in self.mixedDatasetPath.iterdir()]) == (12+24)*2
