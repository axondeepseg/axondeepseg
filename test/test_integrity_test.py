# coding: utf-8

import pytest
from AxonDeepSeg.integrity_test import integrity_test
import AxonDeepSeg.download_tests as download_tests
from pathlib import Path

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.testPath = Path(__file__).resolve().parent
        if not (self.testPath / '__test_files__' ).exists():
            download_tests.main()

        pass

    def teardown_method(self):
        pass

    # --------------integrity_test.py tests-------------- #
    @pytest.mark.integration
    def test_integrity_test_script_runs_successfully(self):
        assert integrity_test() == 0
