# coding: utf-8

from pathlib import Path
import shutil, os
import imageio

import pytest

import AxonDeepSeg
from AxonDeepSeg.download_tests import download_tests


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        print(self.fullPath)
        
        self.package_dir = Path(AxonDeepSeg.__file__).parent  # Get AxonDeepSeg installation path

        # Move up to the test directory, "test/"
        self.testPath = self.package_dir.parent / 'test'

        # Create temp folder
        # Get the directory where this current file is saved
        self.tmpPath = self.testPath / '__tmp__'
        if self.tmpPath.exists():
            shutil.rmtree(self.tmpPath)
            self.tmpPath.mkdir()
        else:
            self.tmpPath.mkdir()
        

        self.temp_test_files = (
            self.tmpPath /
            '__test_files__'
            )



    def teardown_method(self):
        # Get the directory where this current file is saved
        fullPath = Path(__file__).resolve().parent
        print(fullPath)
        # Move up to the test directory, "test/"
        testPath = fullPath.parent 
        tmpPath = testPath / '__tmp__'

        if  (self.testPath / '__test_files__' ).exists() == False:
            (self.testPath / '__test_files__' ).mkdir()
            try: 
                AxonDeepSeg.download_test.main(['-d', self.testPath])
                assert (self.testPath / '__test_files__' ).exists() == True
            except:
                AssertionError('Could not re-download test files, you may need to redownload them or re-install AxonDeepSeg')
        
        if tmpPath.exists():
            shutil.rmtree(tmpPath)
            pass

    # --------------download_model tests-------------- #
    @pytest.mark.integration
    def test_download_tests_runs_succesfully_with_destination(self):
        assert not self.temp_test_files.exists()

        download_tests(self.tmpPath)

        assert self.temp_test_files.exists()

    @pytest.mark.single
    def test_main_cli_runs_succesfully_no_destination(self):

        if self.temp_test_files.exists():
            shutil.rmtree(self.temp_test_files)
            shutil.mkdir(self.temp_test_files)
        dirtree = os.listdir((self.testPath / '__test_files__'))

        for file in dirtree:
            # Make temp dir
            shutil.move(self.testPath / '__test_files__' / file, self.temp_test_files / file)

        shutil.rmtree(self.testPath / '__test_files__')
        assert (self.testPath / '__test_files__').exists() == False

        AxonDeepSeg.download_tests.main([])

        assert (self.testPath / '__test_files__').exists()

        # Remove generated files if it was succesful
        shutil.rmtree(self.testPath / '__test_files__')

        # Move content back
        if not (self.testPath / '__test_files__').exists():
            (self.testPath / '__test_files__').mkdir()

        dirtree = os.listdir(self.temp_test_files)
        for file in dirtree:
            shutil.move(self.temp_test_files / file, self.testPath / '__test_files__' / file)

        shutil.rmtree(self.temp_test_files)

    @pytest.mark.unit
    def test_redownload_test_files_multiple_times_works(self):

        download_tests(self.tmpPath)

    @pytest.mark.unit
    def test_precision_images_expected_types(self):

        precision_path = Path('test/__test_files__/__test_precision_files__')
        rgb_path = Path('test/__test_files__/__test_color_files__')
        rgba_path = Path('test/__test_files__/__test_coloralpha_files__')

        filenames = (
            'image_8bit.png',
            'image_8bit.tif',
            'image_16bit.png',
            'image_16bit.tif',
        )

        expected_dtypes = (
            'uint8',
            'uint8',
            'uint16',
            'uint16',
        )

        for filename, expected_dtype  in zip(filenames, expected_dtypes):
            _img = imageio.v3.imread(precision_path / filename)

            assert str(_img.dtype) == expected_dtype
            assert len(_img.shape) == 2

            if (rgb_path / filename).exists():
                _img = imageio.v3.imread(rgb_path / filename)

                assert str(_img.dtype) == expected_dtype
                assert len(_img.shape) == 3
                assert _img.shape[-1] == 3

            if (rgba_path / filename).exists():
                _img = imageio.v3.imread(rgba_path / filename)

                assert str(_img.dtype) == expected_dtype
                assert len(_img.shape) == 3
                assert _img.shape[-1] == 4
