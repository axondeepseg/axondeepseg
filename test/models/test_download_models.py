# coding: utf-8

from pathlib import Path
import shutil

import pytest

from AxonDeepSeg.models.download_model import *


class TestCore(object):
    def setup(self):
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

        self.sem_model_path = (
            self.tmpPath /
            'default_SEM_model'
            )
        print(self.sem_model_path)

        self.tem_model_path = (
            self.tmpPath /
            'default_TEM_model'
            )

    def teardown(self):
        # Get the directory where this current file is saved
        fullPath = Path(__file__).resolve().parent
        print(fullPath)
        # Move up to the test directory, "test/"
        testPath = fullPath.parent 
        tmpPath = testPath / '__tmp__'

        
        if tmpPath.exists():
            shutil.rmtree(tmpPath)
            pass

    # --------------download_models tests-------------- #
    @pytest.mark.debug
    def test_download_models_works(self):
        assert not self.sem_model_path.exists()
        assert not self.tem_model_path.exists()
        print(self.sem_model_path.absolute())

        download_model(self.tmpPath)

        assert self.sem_model_path.exists()
        assert self.tem_model_path.exists()

    @pytest.mark.debug
    def test_redownload_models_multiple_times_works(self):

        download_model(self.tmpPath)
        download_model(self.tmpPath)
