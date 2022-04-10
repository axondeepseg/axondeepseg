# coding: utf-8

from pathlib import Path

import pytest

from AxonDeepSeg.segment import (
                                    generate_default_parameters,
                                    segment_folders, 
                                    segment_image
                                )
import AxonDeepSeg.segment
import AxonDeepSeg
from config import axonmyelin_suffix, axon_suffix, myelin_suffix

class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.testPath = Path(__file__).resolve().parent
        self.projectPath = self.testPath.parent

        self.modelPath = (
            self.projectPath /
            'AxonDeepSeg' /
            'models' /
            'model_seg_rat_axon-myelin_sem'
            )

        self.modelPathTEM = (
            self.projectPath /
            'AxonDeepSeg' /
            'models' /
            'model_seg_mouse_axon-myelin_tem'
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

        self.imagePathWithPixelSize = self.imageFolderPathWithPixelSize / 'image.png'


        self.imageZoomFolderPathWithPixelSize = (
            self.testPath /
            '__test_files__' /
            '__test_segment_zoom__'
            )
    
        self.imageZoomPathWithPixelSize = self.imageZoomFolderPathWithPixelSize / 'image.png'

        self.imageZoomFolderWithPixelSize = (
            self.testPath /
            '__test_files__' /
            '__test_segment_folder_zoom__'
            )

        self.image16bitTIFGray = (
            self.testPath /
            '__test_files__' /
            '__test_16b_file__' /
            'raw' /
            'data1' /
            'image.tif'
            )

        self.statsFilename = 'model_statistics_validation.json'

    @classmethod
    def teardown_class(cls):

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
            'image.nii.gz',
            'image_2' + str(axon_suffix),
            'image_2' + str(myelin_suffix),
            'image_2' + str(axonmyelin_suffix),
            'image_2.nii.gz'
            ]

        for fileName in outputFiles:

            if (imageFolderPath / fileName).exists():
                (imageFolderPath / fileName).unlink()

            if (imageFolderPathWithPixelSize / fileName).exists():
                (imageFolderPathWithPixelSize / fileName).unlink()

    # --------------segment_folders tests-------------- #
    @pytest.mark.integration
    def test_segment_folders_creates_expected_files(self):
        path_model = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = [48,48]

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix)
            ]

        segment_folders(
            path_testing_images_folder=str(self.imageFolderPath),
            path_model=str(path_model),
            overlap_value=overlap_value,
            acquired_resolution=0.37,
            verbosity_level=2
            )

        for fileName in outputFiles:
            assert (self.imageFolderPath / fileName).exists()

    @pytest.mark.integration
    def test_segment_folders_runs_with_relative_path(self):

        path_model = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = [48,48]

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix)
            ]


        segment_folders(
            path_testing_images_folder=str(self.relativeImageFolderPath),
            path_model=str(path_model),
            overlap_value=overlap_value,
            acquired_resolution=0.37,
            verbosity_level=2
            )

    @pytest.mark.integration
    def test_segment_folders_creates_expected_files_without_acq_res_input(self):
        path_model = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = [48,48]

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix)
            ]

        segment_folders(
            path_testing_images_folder=str(self.imageFolderPathWithPixelSize),
            path_model=str(path_model),
            overlap_value=overlap_value,
            verbosity_level=2
            )

    # --------------segment_image tests-------------- #
    @pytest.mark.integration
    def test_segment_image_creates_runs_successfully(self):
        # Since segment_folders should have already run, the output files
        # should already exist, which this test tests for.

        path_model = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = [48,48]

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix)
            ]

        segment_image(
            path_testing_image=str(self.imagePath),
            path_model=str(path_model),
            overlap_value=overlap_value,
            acquired_resolution=0.37
            )

        for fileName in outputFiles:
            assert (self.imageFolderPath / fileName).exists()

    @pytest.mark.integration
    def test_segment_image_creates_runs_successfully_for_16bit_TIF_gray_file(self):

        path_model = generate_default_parameters('TEM', str(self.modelPathTEM))

        overlap_value = [48,48]

        try:
            segment_image(
                path_testing_image=str(self.image16bitTIFGray),
                path_model=str(path_model),
                overlap_value=overlap_value,
                zoom_factor=1.9
                )
        except:
            pytest.fail("Image segmentation failed for 16bit TIF grayscale file.")

    @pytest.mark.integration
    def test_segment_image_creates_runs_successfully_without_acq_res_input(self):
        # It should work because there exists a pixel file

        path_model = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = [48,48]

        outputFiles = [
            'image' + str(axon_suffix),
            'image' + str(myelin_suffix),
            'image' + str(axonmyelin_suffix)
            ]
        
        segment_image(
            path_testing_image=str(self.imagePathWithPixelSize),
            path_model=str(path_model),
            overlap_value=overlap_value
            )

    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "1", "-s", "0.37"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_with_overlap_value(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "1", "-s", "0.37", '--overlap', '48'])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_with_pixel_size_file(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePathWithPixelSize), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_missing_resolution_size(self):

        # Make sure that the test folder doesn't have a file named pixel_size_in_micrometer.txt
        assert not (self.imageFolderPath / 'pixel_size_in_micrometer.txt').exists()
    
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPath), "-v", "1", "-s", "0.37"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input_with_pixel_size_file(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPathWithPixelSize), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_missing_resolution_size_for_folder_input(self):

        # Make sure that the test folder doesn't have a file named pixel_size_in_micrometer.txt
        assert not (self.imageFolderPath / 'pixel_size_in_micrometer.txt').exists()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)

    @pytest.mark.integration
    def test_main_cli_throws_error_for_too_small_image_without_zoom(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "TEM", "-i", str(self.imageZoomPathWithPixelSize), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 4)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_too_small_image_with_zoom(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "TEM", "-i", str(self.imageZoomPathWithPixelSize), "-v", "1", "-z", "1.2"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_doesnt_throws_error_with_folder_containing_too_small_image_without_zoom(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "TEM", "-i", str(self.imageZoomFolderWithPixelSize), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_folder_containing_too_small_image_with_zoom(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "TEM", "-i", str(self.imageZoomFolderWithPixelSize), "-v", "1", "-z", "1.2"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)
