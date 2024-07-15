# coding: utf-8

from pathlib import Path
import shutil
import tempfile
import os
import pytest

from AxonDeepSeg.segment import (
    segment_folder, 
    segment_images
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
        self.otherImagePath = self.imageFolderPath / 'image_2.png'

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

        sweepFolder = imageFolderPathWithPixelSize / 'image_sweep'

        if sweepFolder.exists():
            shutil.rmtree(sweepFolder)

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
                path_images=str(self.image16bitTIFGray),
                path_model=str(self.modelPath)
            )
        except:
            pytest.fail("Image segmentation failed for 16bit TIF grayscale file.")


    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            # Note that the pixel size set here, 0.1, differs from the true size, 0.37, in order to reduce RAM burden on GitHub Actions CIs and users computers
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "1", "-s", "0.1", "--no-patch", "--overlap", "48"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_creates_logfile(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "1"])

        assert Path('axondeepseg.log').exists()
