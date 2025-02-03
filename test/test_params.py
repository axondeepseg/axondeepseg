# coding: utf-8

import pytest

from ads_base import params


class TestCore(object):
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    # --------------download_data tests-------------- #
    @pytest.mark.unit
    def test_params_returns_expected_binary(self):
        expected_value = 255
        assert params.intensity['binary'] == expected_value

    @pytest.mark.unit
    def test_params_returns_expected_axon(self):
        expected_value = 255
        assert params.intensity['axon'] == expected_value

    @pytest.mark.unit
    def test_params_returns_expected_myelin(self):
        expected_value = 127
        assert params.intensity['myelin'] == expected_value

    @pytest.mark.unit
    def test_params_returns_expected_background(self):
        expected_value = 0
        assert params.intensity['background'] == expected_value
