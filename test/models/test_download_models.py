# coding: utf-8

from pathlib import Path
import shutil

import pytest

from AxonDeepSeg.models.download_model import *


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        print("\n The full path is", self.fullPath)

        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent 
        print("\n The test path is", self.testPath)
        
        # Create temp folder
        # Get the directory where this current file is saved
        self.tmpPath = self.testPath / '__tmp__'
        if not self.tmpPath.exists():
            self.tmpPath.mkdir()
        print("\n The tmp path is ", self.tmpPath)

        self.sem_model_path = (
            self.tmpPath /
            'default_SEM_model'
            )
        print("\n The SEM model path is", self.sem_model_path)

        self.tem_model_path = (
            self.tmpPath /
            'default_TEM_model'
            )

    def teardown(self):
        print("\n Running the teardown method of the test class")
        # Get the directory where this current file is saved
        fullPath = Path(__file__).resolve().parent
        print(fullPath)
        # Move up to the test directory, "test/"
        testPath = fullPath.parent 
        tmpPath = testPath / '__tmp__'

        
        if tmpPath.exists():
            shutil.rmtree(tmpPath)
            pass
        
        #Delete the SEM and TEM model's if present in the test/models directory
        if self.sem_model_path.parent == fullPath:
            shutil.rmtree(self.sem_model_path)
            shutil.rmtree(self.tem_model_path)
        
      

    # --------------download_models tests-------------- #
    @pytest.mark.unit
    def test_download_models_works(self):
        assert not self.sem_model_path.exists()
        assert not self.tem_model_path.exists()
        print(self.sem_model_path.absolute())

        download_model(self.tmpPath)

        assert self.sem_model_path.exists()
        assert self.tem_model_path.exists()

    @pytest.mark.unit
    def test_redownload_models_multiple_times_works(self):

        download_model(self.tmpPath)
        download_model(self.tmpPath)

    @pytest.mark.unit
    def test_download_models_working_directory(self):
        self.sem_model_path = self.fullPath / 'default_SEM_model'
        self.tem_model_path = self.fullPath / 'default_TEM_model'
        download_model(self.fullPath)

        assert self.sem_model_path.parent.name == self.fullPath.name
        assert self.tem_model_path.parent.name == self.fullPath.name

        
