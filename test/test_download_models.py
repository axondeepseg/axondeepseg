# coding: utf-8

from pathlib import Path
import shutil

import pytest

from AxonDeepSeg.download_model import download_model
import AxonDeepSeg
import AxonDeepSeg.download_model


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent 

        self.package_dir = Path(AxonDeepSeg.__file__).parent  # Get AxonDeepSeg installation path
        self.model_dir = self.package_dir / "models"

        self.test_path =  self.package_dir.parent / 'test'
        # Create temp folder
        # Get the directory where this current file is saved
        self.tmp_path = self.test_path / '__tmp__'
        if not self.tmp_path.exists():
            self.tmp_path.mkdir()

        self.valid_model = 'generalist'
        self.valid_model_path = self.tmp_path / 'model_seg_generalist_light'
        self.invalid_model = 'dedicated-BF' # (ensembled version unavailable)
        self.invalid_model_type = 'ensemble'

    def teardown_method(self):
        # If tmp/models folder isn't empty, move it back
        dirtree = list((self.tmp_path / "models").iterdir())
        if (self.tmp_path / "models").exists() & (not self.model_dir.exists()):
            for file in dirtree:
                shutil.move(file, self.model_dir)
                shutil.rmtree(self.tmp_path / "models")
        
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)

    # --------------download_model tests-------------- #
    @pytest.mark.unit
    def test_download_valid_model_works(self):

        assert not self.valid_model_path.exists()
        download_model(self.valid_model, 'light', self.tmp_path)
        assert self.valid_model_path.exists()

    @pytest.mark.unit
    def test_download_model_cli_throws_error_for_unavailable_model(self):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(
                ["-m", self.invalid_model, "-t", self.invalid_model_type]
            )

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 1)


    @pytest.mark.unit
    def test_redownload_model_multiple_times_works(self):

        download_model(self.valid_model, 'light', self.tmp_path)
        download_model(self.valid_model, 'light', self.tmp_path)
        
    @pytest.mark.unit
    def test_list_models(self):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(["-l"])
        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_for_list_models(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(["--list"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integratikon
    def test_main_cli_runs_succesfully_no_destination(self):
        # Move content inside model folder to test/__test_files__/tmp_folder, without moving the folder itself
        if not (self.tmp_path / "models").exists():
            (self.tmp_path / "models").mkdir()
        
        dirtree = list(self.model_dir.iterdir())
        for file in dirtree:
            # Make temp dir
            shutil.move(file, self.tmp_path / "models")

        AxonDeepSeg.download_model.main([])
        assert (self.package_dir / 'models/model_seg_generalist_light').exists()

        # Move content back
        dirtree = list((self.tmp_path / "models").iterdir())
        for file in dirtree:
            shutil.rmtree(self.tmp_path / "models")

    @pytest.mark.integration
    def test_main_cli_downloads_to_path(self):
        cli_test_path = self.tmp_path / 'cli_test'
        cli_test_model_path = cli_test_path / 'model_seg_generalist_light'

        AxonDeepSeg.download_model.main(["-d", str(cli_test_path)])

        assert cli_test_model_path.exists()


    @pytest.mark.integration
    def test_main_cli_fails_for_model_that_does_not_exist(self):
        model_name = "no_model"
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(["-m ", model_name])

        expected_code = 2
        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == expected_code)
