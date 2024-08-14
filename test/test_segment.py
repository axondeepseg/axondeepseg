# coding: utf-8

from pathlib import Path
import shutil
import tempfile
import os
import pytest

from AxonDeepSeg.segment import (
    segment_folder, 
    segment_images,
    get_model_type
)
import AxonDeepSeg
from config import axonmyelin_suffix, axon_suffix, myelin_suffix

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.testPath = Path(__file__).resolve().parent
        self.projectPath = self.testPath.parent

        self.modelPath = (
            self.projectPath /
            'AxonDeepSeg' /
            'models' /
            'model_seg_generalist_light'
            )

        self.imageFolderPath = (
            self.testPath /
            '__test_files__' /
            '__test_segment_files__'
            )
        self.relativeImageFolderPath = Path(
            'test/__test_files__/__test_segment_files__'
        )

        self.imagePath = self.imageFolderPath / 'image.png'

        self.imageFolderPathWithPixelSize = (
            self.testPath /
            '__test_files__' /
            '__test_segment_files_with_pixel_size__'
            )

        self.image16bit_folder = (
            self.testPath /
            '__test_files__' /
            '__test_16b_file__' /
            'raw' /
            'data1'
        )

        self.image16bitTIFGray = (
            self.image16bit_folder /
            'image.tif'
        )

        self.expected_image_16bit_output_files = [
            self.image16bit_folder / ('image' + str(axon_suffix)),
            self.image16bit_folder / ('image' + str(myelin_suffix)),
            self.image16bit_folder / ('image' + str(axonmyelin_suffix)),
            self.image16bit_folder / 'image.png',    # TIF image should be converted to PNG
        ]

        self.nnunetModelLight = (
            self.projectPath /
            'AxonDeepSeg' /
            'models' /
            'model_seg_generalist_light'
        )

        self.nnunetModelEmptyEnsemble = (
            self.projectPath /
            'test' /
            '__test_files__' /
            '__test_model__' /
            'models' / 
            'model_empty_ensemble'
            )

    def teardown_method(self):

        testPath = Path(__file__).resolve().parent
        projectPath = testPath.parent
        imageFolderPath = testPath / '__test_files__' / '__test_segment_files__'

        imageFolderPathWithPixelSize = (
            testPath /
            '__test_files__' /
            '__test_segment_files_with_pixel_size__'
            )

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix),
            'image_2_grayscale.png',
            'image_2_grayscale' + str(axon_suffix),
            'image_2_grayscale' + str(myelin_suffix),
            'image_2_grayscale' + str(axonmyelin_suffix)
            ]

        logfile = testPath / 'axondeepseg.log'

        for fileName in outputFiles:

            if (imageFolderPath / fileName).exists():
                (imageFolderPath / fileName).unlink()

            if (imageFolderPathWithPixelSize / fileName).exists():
                (imageFolderPathWithPixelSize / fileName).unlink()
        
        if logfile.exists():
            logfile.unlink()

        for output_16bit in self.expected_image_16bit_output_files:
            if output_16bit.exists():
                output_16bit.unlink()

    # --------------segment_folder tests-------------- #
    @pytest.mark.unit
    def test_segment_folder_creates_expected_files(self):

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix),
            'image_2_grayscale.png',
            'image_2_grayscale' + str(axon_suffix),
            'image_2_grayscale' + str(myelin_suffix),
            'image_2_grayscale' + str(axonmyelin_suffix)
            ]

        segment_folder(
            path_folder=str(self.imageFolderPath),
            path_model=str(self.modelPath),
            verbosity_level=2
        )

        for fileName in outputFiles:
            assert (self.imageFolderPath / fileName).exists()

    @pytest.mark.unit
    def test_segment_folder_runs_with_relative_path(self):

        segment_folder(
            path_folder=str(self.relativeImageFolderPath),
            path_model=str(self.modelPath),
            verbosity_level=2
        )


    # --------------segment_image tests-------------- #
    @pytest.mark.unit
    def test_segment_images_creates_expected_files(self):

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix),
            ]

        segment_images(
            path_images=[str(self.imagePath)],
            path_model=str(self.modelPath),
        )

        for fileName in outputFiles:
            assert (self.imageFolderPath / fileName).exists()

    @pytest.mark.unit
    def test_segment_image_creates_runs_successfully_for_16bit_TIF_gray_file(self):

        try:
            segment_images(
                path_images=[str(self.image16bitTIFGray)],
                path_model=str(self.modelPath)
            )
            
        except:
            pytest.fail("Image segmentation failed for 16bit TIF grayscale file.")
        
        for out_file in self.expected_image_16bit_output_files:
            assert out_file.exists()

    # --------------get_model_type tests-------------- #
    @pytest.mark.unit
    def test_get_model_type_light(self):
        path_model = self.nnunetModelLight
        model_type = get_model_type(path_model)
        expected_model_type = 'light'

        assert model_type == expected_model_type

    @pytest.mark.unit
    def test_get_model_type_ensemble(self):
        path_model = self.nnunetModelEmptyEnsemble
        model_type = get_model_type(path_model)
        expected_model_type = 'ensemble'

        assert model_type == expected_model_type


    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-i", str(self.imagePath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-i", str(self.imageFolderPath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_creates_logfile(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-i", str(self.imagePath), "-v", "1"])

        assert Path('axondeepseg.log').exists()


