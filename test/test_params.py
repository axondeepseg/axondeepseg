# coding: utf-8

import pytest
from pathlib import Path

from AxonDeepSeg import params


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

    @pytest.mark.unit
    def test_all_output_suffixes_excluded_from_folder_segmentation(self):
        """
        All known pipeline-generated file suffixes must be present in generated_file_suffixes.
        The segmenter uses generated_file_suffixes to skip outputs when batch-processing a
        folder; a missing suffix causes the segmenter to attempt to re-segment its own outputs.
        If a new output suffix is added to params.py, add it here too.
        """
        expected = [
            params.axonmyelin_suffix,
            params.axon_suffix,
            params.myelin_suffix,
            params.index_suffix,
            params.axonmyelin_index_suffix,
            params.unmyelinated_suffix,
            params.unmyelinated_index_suffix,
            params.nnunet_suffix,
            params.nerve_suffix,
            params.nerve_index_suffix,
            params.diameter_overlay_suffix,
            params.instance_im_suffix,
            params.instance_suffix,
            params.morph_suffix,
            params.morph_agg_suffix,
            params.unmyelinated_morph_suffix,
            params.nerve_morph_suffix,
        ]
        missing = [
            str(s) for s in expected
            if not str(s).endswith(params.generated_file_suffixes)
        ]
        assert missing == [], f"These output suffixes are not in generated_file_suffixes: {missing}"
