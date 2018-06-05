# coding: utf-8

import pytest
from AxonDeepSeg.integrity_test import integrity_test

class TestCore(object):
    def setup(self):
        pass
    def teardown(self):
        pass

    #--------------integrity_test.py tests--------------#
    @pytest.mark.integritytest
    def test_integrity_test_script_runs_succesfully(self):
        assert integrity_test()==0
