# coding: utf-8

from pathlib import Path
import shutil
import os

import pytest

from AxonDeepSeg.download_model import download_model, main


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

        self.sem_model_path = (
            self.tmpPath /
            'model_seg_rat_axon-myelin_sem'
            )
        self.tem_model_path = (
            self.tmpPath /
            'model_seg_mouse_axon-myelin_tem'
            )
        self.bf_model_path = (
            self.tmpPath /
            'model_seg_rat_axon-myelin_bf'
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
        
        #Delete the SEM and TEM model's if present in the test/models directory
        if self.sem_model_path.parent == fullPath:
            shutil.rmtree(self.sem_model_path)
            shutil.rmtree(self.tem_model_path)
            shutil.rmtree(self.bf_model_path)

    # --------------download_models tests-------------- #
    @pytest.mark.single
    def test_download_models_works(self):
        assert not self.sem_model_path.exists()
        assert not self.tem_model_path.exists()
        assert not self.bf_model_path.exists()

        download_model(self.tmpPath)

        assert self.sem_model_path.exists()
        assert self.tem_model_path.exists()
        assert self.bf_model_path.exists()

    @pytest.mark.single
    def test_redownload_models_multiple_times_works(self):

        download_model(self.tmpPath)
        download_model(self.tmpPath)
        
    @pytest.mark.single
    def test_download_models_working_directory(self):
        self.sem_model_path = self.fullPath / "model_seg_rat_axon-myelin_sem"
        self.tem_model_path = self.fullPath / "model_seg_mouse_axon-myelin_tem"
        self.bf_model_path = self.fullPath / "model_seg_rat_axon-myelin_bf"
        download_model(self.fullPath)

        assert self.sem_model_path.exists()
        assert self.tem_model_path.exists()
        assert self.bf_model_path.exists()


    # --------------download_models (cli) test-------------- #
    @pytest.mark.single
    def test_download_models_cli_runs_successfully_with_valid_destination_folder(self):
        destination_path = self.fullPath    # set the destination path to test/models
        self.sem_model_path = self.fullPath / "model_seg_rat_axon-myelin_sem"
        self.tem_model_path = self.fullPath / "model_seg_mouse_axon-myelin_tem"
        self.bf_model_path = self.fullPath / "model_seg_rat_axon-myelin_bf"

        main(["-p", str(destination_path)])
        assert self.sem_model_path.exists()
        assert self.tem_model_path.exists()
        assert self.bf_model_path.exists()