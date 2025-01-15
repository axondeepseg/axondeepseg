# coding: utf-8

from pathlib import Path
import shutil
import numpy as np

import pytest

from AxonDeepSeg.ads_utils import download_data, convert_path, get_existing_models_list, extract_axon_and_myelin_masks_from_image_data, imread, get_file_extension, check_available_gpus
from AxonDeepSeg.model_cards import get_supported_models
from AxonDeepSeg import params
from torch.cuda import device_count

class TestCore(object):
    def setup_method(self):
        self.osf_link = "https://osf.io/uw5hv/?action=download"
        
        self.bad_osf_link = "https://af7lafuoDs"

        self.precision_path = Path('test/__test_files__/__test_precision_files__')

    def teardown_method(self):
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
        known_models = get_supported_models()

        for downloaded_model in get_existing_models_list():
            assert downloaded_model in known_models

    @pytest.mark.unit
    def test_imread_fails_for_ome_filename(self):
        filename = 'test_name.ome.tif'

        with pytest.raises(IOError):
            imread(filename)

    @pytest.mark.unit
    def test_imread_same_output_for_different_input_precisions(self):
        filenames = {
            'image_8bit_int.png',
            'image_16bit_int.png',
            'image_32bit_int.png',
            'image_16bit_float.png',
            'image_32bit_float.png',
        }

        image_1 = None
        image_2 = None
        for file in filenames:
            print(file)
            image = (imread(self.precision_path / file))\
            
            if image_1 is None:
                image_1 = image
            else:
                image_2 = image

                assert np.allclose(image_1, image_2, atol=1) # In very few pixels (eg, 3 pixels in entire image), rounding differences between float and int conversions lead to an int difference value of 1, which is why this atol was chosen.

    @pytest.mark.unit
    def test_get_file_extension_returns_expected_filenames(self):
        filenames_lowercase = [
                    'test_name.jpg',
                    'test_name.jpeg',
                    'test_name.png',
                    'test_name.tif',
                    'test_name.tiff',
                    'test_name.ome.tif'
        ]
        expected_extensions = [
                    '.jpg',
                    '.jpeg',
                    '.png',
                    '.tif',
                    '.tiff',
                    '.ome.tif'
        ]

        filenames_uppercase = [
                    'test_name.JPG',
                    'test_name.JPEG',
                    'test_name.PNG',
                    'test_name.TIF',
                    'test_name.TIFF',
                    'test_name.OME.TIF'
        ]

        for filename, ext in zip(filenames_lowercase, expected_extensions):
            assert get_file_extension(filename) == ext
        
        for filename, ext in zip(filenames_uppercase, expected_extensions):
            assert get_file_extension(filename) == ext

    @pytest.mark.unit
    def test_get_file_extension_works_with_periods_inside_filenames(self):
        filenames = [
            'something_seg-axon_zf-1.25.png',
            'something_seg-axon_zf-1.25.ome.tiff',
        ]

        expected_extensions = [
            '.png',
            '.ome.tiff'
        ]

        for filename, ext in zip(filenames, expected_extensions):
            assert get_file_extension(filename) == ext

    @pytest.mark.unit
    def test_check_available_gpus(self):
        gpu_id = 0
        n_gpus = check_available_gpus(gpu_id)

        expected_n_gpus = device_count()

        assert n_gpus == expected_n_gpus
