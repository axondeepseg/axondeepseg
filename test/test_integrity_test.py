# coding: utf-8

import pytest
from ads_base.integrity_test import integrity_test


class TestCore(object):
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    # --------------integrity_test.py tests-------------- #
    @pytest.mark.integration
    def test_integrity_test_script_runs_successfully(self):
        assert integrity_test() == 0
