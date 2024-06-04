# coding: utf-8

from pathlib import Path
import shutil

import pytest

from AxonDeepSeg.download_model import download_model
import AxonDeepSeg


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

        self.valid_model = 'generalist'
        self.valid_model_path = self.tmpPath / 'model_seg_generalist_light'
        self.invalid_model = 'dedicated-BF' # (ensembled version unavailable)

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

    # --------------download_models tests-------------- #
    @pytest.mark.unit
    def test_download_valid_model_works(self):

        assert not self.valid_model_path.exists()
        download_model(self.valid_model, 'light', self.tmpPath)
        assert self.valid_model_path.exists()

    @pytest.mark.unit
    def test_download_model_cli_throws_error_for_unavailable_model(self):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(
                ["-m", self.invalid_model, "-t", "ensemble"]
            )

        assert pytest_wrapped_e.type == SystemExit


    @pytest.mark.unit
    def test_redownload_models_multiple_times_works(self):

        download_model(self.valid_model, 'light', self.tmpPath)
        download_model(self.valid_model, 'light', self.tmpPath)
        
