# coding: utf-8
import AxonDeepSeg
from pathlib import Path
import shutil
import imageio

import pytest

from AxonDeepSeg.download_tests import download_tests

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent

        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent 

        # Create temp folder
        # Get the directory where this current file is saved
        self.tmpPath = self.testPath / '__tmp__'
        if not self.tmpPath.exists():
            self.tmpPath.mkdir()
        print(self.tmpPath)

        self.test_files_path = (
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

        
        if tmpPath.exists():
            shutil.rmtree(tmpPath)
            pass

    # --------------download_model tests-------------- #
    @pytest.mark.unit
    def test_download_tests_works(self):
        assert not self.test_files_path.exists()

        download_tests(self.tmpPath)

        assert self.test_files_path.exists()
        
    @pytest.mark.integration
    def test_download_tests_runs_succesfully_with_destination(self):
        assert not self.test_files_path.exists()

        download_tests(self.tmpPath)
 
        assert self.test_files_path.exists()

        # Remove it
        shutil.rmtree(self.test_files_path)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_no_destination(self):
        cli_test_model_path =  Path(AxonDeepSeg.__file__).parent.parent / 'test' / '__test_files__'

        output_dir = download_tests(destination=None, overwrite=False)

        assert output_dir == cli_test_model_path


    @pytest.mark.unit
    def test_redownload_test_files_multiple_times_works(self):
        assert not self.test_files_path.exists()
        download_tests(self.tmpPath)
        download_tests(self.tmpPath)

        # Remove it
        shutil.rmtree(self.test_files_path)

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
