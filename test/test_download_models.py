# coding: utf-8

from pathlib import Path
import shutil

import pytest

from AxonDeepSeg.download_model import download_model


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

        self.ivado_model_path = (
            self.tmpPath /
            'default-SEM-model'
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
    @pytest.mark.unit
    def test_download_models_works(self):
        assert not self.ivado_model_path.exists()

        download_model(self.tmpPath)

        assert self.ivado_model_path.exists()


    @pytest.mark.unit
    def test_redownload_models_multiple_times_works(self):

        download_model(self.tmpPath)
        download_model(self.tmpPath)
        
