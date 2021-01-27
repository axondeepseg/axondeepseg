# coding: utf-8

from pathlib import Path

import pytest

from AxonDeepSeg.segment import *
import AxonDeepSeg.segment
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
            'default_SEM_model'
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
            'image' + axon_suffix.name,
            'image' + myelin_suffix.name,
            'image' + axonmyelin_suffix.name,
            'image2' + axon_suffix.name,
            'image2' + myelin_suffix.name,
            'image2' + axonmyelin_suffix.name
            ]

        for fileName in outputFiles:
            if (imageFolderPath / fileName).exists():
                (imageFolderPath / fileName).unlink()

            if (imageFolderPathWithPixelSize / fileName).exists():
                (imageFolderPathWithPixelSize / fileName).unlink()

    # --------------generate_config_dict tests-------------- #
    @pytest.mark.unit
    def test_generate_config_dict_outputs_dict(self):

        config =  generate_config_dict(str(self.modelPath / 'config_network.json'))

        assert type(config) is dict

    @pytest.mark.unit
    def test_generate_config_dict_throws_exception_for_nonexisting_file(self):

        with pytest.raises(ValueError):
            config = generate_config_dict(str(self.modelPath / 'n0n_3xist1ng_f1l3.json'))

    # --------------generate_resolution tests-------------- #
    @pytest.mark.unit
    def test_generate_resolution_returns_expected_known_project_cases(self):

        assert generate_resolution('SEM', 512) == 0.1
        assert generate_resolution('SEM', 256) == 0.2
        assert generate_resolution('TEM', 512) == 0.01

    # --------------segment_folders tests-------------- #
    @pytest.mark.integration
    def test_segment_folders_creates_expected_files(self):

        path_model, config = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = 25
        resolution_model = generate_resolution('SEM', 512)

        outputFiles = [
            'image' + axon_suffix.name,
            'image' + myelin_suffix.name,
            'image' + axonmyelin_suffix.name
            ]

        for fileName in outputFiles:
            assert not (self.imageFolderPath / fileName).exists()

        segment_folders(
            path_testing_images_folder=str(self.imageFolderPath),
            path_model=str(path_model),
            overlap_value=overlap_value,
            config=config,
            resolution_model=resolution_model,
            acquired_resolution=0.37,
            verbosity_level=2
            )

        for fileName in outputFiles:
            assert (self.imageFolderPath / fileName).exists()

    @pytest.mark.integration
    def test_segment_folders_runs_with_relative_path(self):

        path_model, config = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = 25
        resolution_model = generate_resolution('SEM', 512)

        outputFiles = [
            'image' + axon_suffix.name,
            'image' + myelin_suffix.name,
            'image' + axonmyelin_suffix.name
            ]

        segment_folders(
            path_testing_images_folder=str(self.relativeImageFolderPath),
            path_model=str(path_model),
            overlap_value=overlap_value,
            config=config,
            resolution_model=resolution_model,
            acquired_resolution=0.37,
            verbosity_level=2
            )

    # --------------segment_image tests-------------- #
    @pytest.mark.integration
    def test_segment_image_creates_runs_successfully(self):
        # Since segment_folders should have already run, the output files
        # should already exist, which this test tests for.

        path_model, config = generate_default_parameters('SEM', str(self.modelPath))

        overlap_value = 25
        resolution_model = generate_resolution('SEM', 512)

        outputFiles = [
            'image' + axon_suffix.name,
            'image' + myelin_suffix.name,
            'image' + axonmyelin_suffix.name
            ]

        for fileName in outputFiles:
            assert (self.imageFolderPath / fileName).exists()

        segment_image(
            path_testing_image=str(self.imagePath),
            path_model=str(path_model),
            overlap_value=overlap_value,
            config=config,
            resolution_model=resolution_model,
            acquired_resolution=0.37,
            verbosity_level=2
            )

        for fileName in outputFiles:
            assert (self.imageFolderPath / fileName).exists()

    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "2", "-s", "0.37"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_with_pixel_size_file(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePathWithPixelSize), "-v", "2"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_for_too_small_resolution_due_to_min_resampled_patch_size(self):

        image_size = [436, 344] # of self.imagePath
        default_SEM_resolution = 0.1
        default_SEM_patch_size = 512

        minimum_resolution = default_SEM_patch_size * default_SEM_resolution / min(image_size)

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "2", "-s", str(round(0.99*minimum_resolution,3))])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 2)

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_missing_resolution_size(self):

        # Make sure that the test folder doesn't have a file named pixel_size_in_micrometer.txt
        assert not (self.imageFolderPath / 'pixel_size_in_micrometer.txt').exists()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imagePath), "-v", "2"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPath), "-v", "2", "-s", "0.37"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input_with_pixel_size_file(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPathWithPixelSize), "-v", "2"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_for_too_small_resolution_due_to_min_resampled_patch_size_for_folder_input(self):

        image_size = [436, 344] # of self.imagePath
        default_SEM_resolution = 0.1
        default_SEM_patch_size = 512

        minimum_resolution = default_SEM_patch_size * default_SEM_resolution / min(image_size)

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPath), "-v", "2", "-s", str(round(0.99*minimum_resolution,3))])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 2)

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_missing_resolution_size_for_folder_input(self):

        # Make sure that the test folder doesn't have a file named pixel_size_in_micrometer.txt
        assert not (self.imageFolderPath / 'pixel_size_in_micrometer.txt').exists()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(self.imageFolderPath), "-v", "2"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)
