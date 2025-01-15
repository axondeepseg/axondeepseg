# coding: utf-8

from pathlib import Path
import shutil

import pytest

from AxonDeepSeg.download_tests import download_tests


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        print(self.fullPath)
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

    @pytest.mark.unit
    def test_redownload_test_files_multiple_times_works(self):

        download_tests(self.tmpPath)
