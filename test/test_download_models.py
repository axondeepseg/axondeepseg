# coding: utf-8

from pathlib import Path
import shutil

import pytest

from ads_base.download_model import download_model
import ads_base
import ads_base.download_model


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
        self.invalid_model_type = 'ensemble'

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
    def test_download_valid_model_works(self):

        assert not self.valid_model_path.exists()
        download_model(self.valid_model, 'light', self.tmpPath)
        assert self.valid_model_path.exists()

    @pytest.mark.unit
    def test_download_model_cli_throws_error_for_unavailable_model(self):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.download_model.main(
                ["-m", self.invalid_model, "-t", self.invalid_model_type]
            )

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 1)


    @pytest.mark.unit
    def test_redownload_model_multiple_times_works(self):

        download_model(self.valid_model, 'light', self.tmpPath)
        download_model(self.valid_model, 'light', self.tmpPath)
        
    @pytest.mark.unit
    def test_list_models(self):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.download_model.main(["-l"])
        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_for_list_models(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.download_model.main(["--list"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_no_destination(self):
        cli_test_model_path = Path('.') / 'models' / 'model_seg_generalist_light'

        ads_base.download_model.main(["-t","light"])

        assert cli_test_model_path.exists()

    @pytest.mark.integration
    def test_main_cli_downloads_to_path(self):
        cli_test_path = self.tmpPath / 'cli_test'
        cli_test_model_path = cli_test_path / 'model_seg_generalist_light'

        ads_base.download_model.main(["-d", str(cli_test_path)])

        assert cli_test_model_path.exists()


    @pytest.mark.integration
    def test_main_cli_fails_for_model_that_does_not_exist(self):
        model_name = "no_model"
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.download_model.main(["-m ", model_name])

        expected_code = 2
        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == expected_code)
