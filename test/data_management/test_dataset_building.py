# coding: utf-8

import imageio
from pathlib import Path
import shutil
import pytest
from AxonDeepSeg.data_management.dataset_building import *
from AxonDeepSeg.visualization.get_masks import *


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        _create_new_test_folder = (
            lambda s, t: self.testPath
            / '__test_files__'
            / s
            / t
            )

        self.rawPath =  _create_new_test_folder('__test_patch_files__', 'raw')
        self.patchPath =  _create_new_test_folder('__test_patch_files__', 'patched')
        self.datasetPath = _create_new_test_folder('__test_patch_files__', 'dataset')
        self.mixedPatchPath = _create_new_test_folder('__test_patch_files__', 'mixedPatched')
        self.mixedDatasetPath = _create_new_test_folder('__test_patch_files__', 'mixedDataset')

        self.rawPath16b =   _create_new_test_folder('__test_16b_file__', 'raw')
        self.patchPath16b =   _create_new_test_folder('__test_16b_file__', 'patched')

    @classmethod
    def teardown_class(cls):
        # Get the directory where this current file is saved
        fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        testPath = fullPath.parent

        _create_new_test_folder = (
            lambda s, t: testPath
            / '__test_files__'
            / s
            / t
            )

        patchPath =  _create_new_test_folder('__test_patch_files__', 'patched')
        datasetPath = _create_new_test_folder('__test_patch_files__', 'dataset')
        mixedPatchPath = _create_new_test_folder('__test_patch_files__', 'mixedPatched')
        mixedDatasetPath = _create_new_test_folder('__test_patch_files__', 'mixedDataset')

        patchPath16b =   _create_new_test_folder('__test_16b_file__', 'patched')

        if patchPath.is_dir():
            shutil.rmtree(patchPath)
 
        if datasetPath.is_dir():
            shutil.rmtree(datasetPath)
 
        if mixedPatchPath.is_dir():
            shutil.rmtree(mixedPatchPath)
 
        if mixedDatasetPath.is_dir():
            shutil.rmtree(mixedDatasetPath)

        if patchPath16b.is_dir():
            shutil.rmtree(patchPath16b)

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

    @pytest.mark.unit
    def test_raw_img_to_patches_doesnt_cutoff_16bit_files(self):
        if self.patchPath16b.is_dir():
            shutil.rmtree(self.patchPath16b)

        raw_img_to_patches(str(self.rawPath16b), str(self.patchPath16b), patch_size=512, resampling_resolution=0.005)

        img_folder_names = [im.name for im in self.patchPath16b.iterdir()]
        for img_folder in tqdm(img_folder_names):
            path_img_folder = self.patchPath16b / img_folder

            if path_img_folder.is_dir():
                # We go through every file in the image folder
                data_names = [d.name for d in path_img_folder.iterdir()]
                for data in data_names:
                    # Skip the mask files
                    if 'mask' not in data:
                        print(data)
                        img = imageio.imread(path_img_folder / data)
                        img_bins = np.bincount(np.ndarray.flatten(img))
                    
                        # Assert that not more than 50% of the pixels are the minimum value
                        assert img_bins[0]/sum(img_bins) < 0.5

                        # Assert that not more than 50% of the pixels are the maximum value
                        assert img_bins[-1]/sum(img_bins) < 0.5

    @pytest.mark.unit
    def test_raw_img_to_patches_creates_masks_with_expected_number_of_unique_values(self):
        if self.patchPath.is_dir():
            shutil.rmtree(self.patchPath)

        raw_img_to_patches(str(self.rawPath), str(self.patchPath))

        
        img_folder_names = [im.name for im in self.patchPath.iterdir()]
        for img_folder in tqdm(img_folder_names):
            path_img_folder = self.patchPath / img_folder
            if path_img_folder.is_dir():
                # We go through every file in the image folder
                data_names = [d.name for d in path_img_folder.iterdir()]
                for data in data_names:

                    if 'mask' in data:
                        mask = imageio.imread(path_img_folder / data)
                        
                        image_properties = get_image_unique_vals_properties(mask)

                        assert image_properties['num_uniques'] == 3
                        assert np.array_equal(image_properties['unique_values'], [0, 128, 255])

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
