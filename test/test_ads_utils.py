# coding: utf-8

import pytest
import os
import shutil
from pathlib import Path
from AxonDeepSeg.ads_utils import download_data


class TestCore(object):
    def setup(self):
        self.osf_link = "https://osf.io/uw5hv/?action=download"
        
        self.bad_osf_link = "https://af7lafuoDs"

    def teardown(self):
        output_path = "TEM_striatum" # Name of zip file downloaded, folder was created in this name too
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

    # --------------download_data tests-------------- #
    @pytest.mark.unit
    def test_download_data_returns_0_for_valid_link(self):
        exit_code = download_data(self.osf_link)
        assert exit_code == 0

    @pytest.mark.unit
    def test_download_data_returns_1_for_invalid_link(self):
        exit_code = download_data(self.bad_osf_link)
        assert exit_code == 1

