# coding: utf-8

import pytest
import os
import shutil
from AxonDeepSeg.ads_utils import download_osf


class TestCore(object):
    def setup(self):
        self.osf_link = "https://osf.io/uw5hv/?action=download"
        self.zip_filename = "TEM_striatum.zip"
        
        self.bad_osf_link = "https://af7lafuoDs"
        self.bad_zip_filename = "NONEXISTANT_TEM_striatum.zip"


    def teardown(self):
        output_path = os.path.splitext(self.zip_filename)
        if os.path.exists(output_path[0]):
            shutil.rmtree(output_path[0])

    # --------------download_osf tests-------------- #
    @pytest.mark.unit
    def test_download_osf_returns_0_for_valid_link(self):
        exit_code = download_osf(self.osf_link, self.zip_filename)
        assert exit_code == 0

    @pytest.mark.unit
    def test_download_osf_returns_1_for_invalid_link(self):
        exit_code = download_osf(self.bad_osf_link, self.zip_filename)
        assert exit_code == 1

