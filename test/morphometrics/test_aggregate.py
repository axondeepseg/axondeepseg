# coding: utf-8

from pathlib import Path
import string
import random
import math
import shutil
import numpy as np
from AxonDeepSeg import ads_utils as ads
import AxonDeepSeg
import pytest

# AxonDeepSeg imports
from AxonDeepSeg.morphometrics.aggregate import *

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.test_folder_path = (
            self.testPath /
            '__test_files__' /
            '__test_aggregate__' / 
            'all_subjects'
            )
        

    def teardown_method(self):
        pass


    # --------------get_pixelsize tests-------------- #
    @pytest.mark.integration
    def test_aggregate_cli_runs_and_produces_expected_directories(self):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.aggregate.main(["-i", str(self.test_folder_path)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)
        assert (Path(self.test_folder_path) / "morphometrics_agg").exists()
        assert (Path(self.test_folder_path) / "morphometrics_agg" / "subject 1").exists()
        assert (Path(self.test_folder_path) / "morphometrics_agg" / "subject 2").exists()
    
    @pytest.mark.integration
    def test_aggregate_load_morphometrics_empty_file(self):
        # Load excel file
        df = pd.read_excel(self.testPath / '__test_files__' / '__test_noaxons_files__' / 'image_axon_morphometrics.xlsx')
        metrics_df = load_morphometrics(df, filters={"gratio_null": True, "gratio_sup": True})

        assert metrics_df
            
