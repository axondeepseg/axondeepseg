# coding: utf-8

from pathlib import Path
import shutil
import numpy as np

import pytest

from AxonDeepSeg.ads_utils import download_data, convert_path, get_existing_models_list, extract_axon_and_myelin_masks_from_image_data
from AxonDeepSeg import params


class TestCore(object):
    def setup(self):
        self.osf_link = "https://osf.io/uw5hv/?action=download"
        
        self.bad_osf_link = "https://af7lafuoDs"

    def teardown(self):
        output_path = Path("TEM_striatum") # Name of zip file downloaded, folder was created in this name too
        if output_path.exists():
            shutil.rmtree(output_path)

    # --------------download_data tests-------------- #
    @pytest.mark.unit
    def test_download_data_returns_0_for_valid_link(self):
        exit_code = download_data(str(self.osf_link))
        assert exit_code == 0

    @pytest.mark.unit
    def test_download_data_returns_1_for_invalid_link(self):
        exit_code = download_data(str(self.bad_osf_link))
        assert exit_code == 1

    # --------------convert_path tests-------------- #
    @pytest.mark.unit
    def test_convert_path_Path_object_input_returns_Path_object(self):
        object_path = Path('folder_name/')
        object_path = convert_path(object_path)

        assert isinstance(object_path, Path)

    @pytest.mark.unit
    def test_convert_path_str_object_input_returns_Path_object(self):
        object_path = 'folder_name/'
        object_path = convert_path(object_path)

        assert isinstance(object_path, Path)

    @pytest.mark.unit
    def test_convert_path_None_returns_None(self):
        object_path = None
        object_path = convert_path(object_path)

        assert object_path == None

    @pytest.mark.unit
    def test_convert_path_unexpected_input_raises_TypeError(self):
        object_path = 2019

        with pytest.raises(TypeError):
            object_path = convert_path(object_path)

    @pytest.mark.unit
    def test_convert_list_of_str_returns_expect_list(self):
        object_path = ['folder_name/', Path('folder_name/'), None]
        object_path = convert_path(object_path)

        expected_output = [Path('folder_name/').absolute(), Path('folder_name/').absolute(), None]
        assert expected_output == object_path

    @pytest.mark.unit
    def test_extract_data_returns_expected_arrays(self):
        image_data = np.zeros((3, 3), dtype=np.uint8)
        image_data[1, :] = params.intensity['myelin']
        image_data[2, :] = params.intensity['axon']

        expected_axon_array = (image_data == params.intensity['axon']).astype(np.uint8)
        expected_myelin_array = (image_data == params.intensity['myelin']).astype(np.uint8)

        obtained_axon_array, obtained_myelin_array = extract_axon_and_myelin_masks_from_image_data(image_data)

        assert np.array_equal(expected_axon_array, obtained_axon_array)
        assert np.array_equal(expected_myelin_array, obtained_myelin_array)

    @pytest.mark.unit
    def test_get_existing_models_list_returns_known_models(self):
        known_models = ['default_TEM_model', 'default_SEM_model']

        for model in known_models:
            assert model in get_existing_models_list()
