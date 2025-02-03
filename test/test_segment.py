# coding: utf-8

from pathlib import Path
import pytest

from ads_base.segment import (
    segment_folder, 
    segment_images,
    get_model_type,
    prepare_inputs
)
import ads_base
from ads_base.params import axonmyelin_suffix, axon_suffix, myelin_suffix

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.testPath = Path(__file__).resolve().parent
        self.projectPath = self.testPath.parent

        self.modelPath = (
            self.projectPath /
            'ads_base' /
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
            'ads_base' /
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

        self.to_delete = []

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

        logfile = testPath / 'ads_base.log'

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
        
        for file in self.to_delete:
            file.unlink()

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


    @pytest.mark.unit
    def test_segment_image_exits_for_nonexistent_file(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.segment.main(["-i", str(Path('/image/does/not/exist/image.png'))])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 2)

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

    # --------------prepare_inputs tests-------------- #
    @pytest.mark.unit
    def test_prepare_inputs_grayscale_to_1channel_valid(self):
        path_imgs = [self.testPath / '__test_files__'/ '__test_demo_files__' / 'image.png']
        file_format = '.png'
        n_channels = 1
        prepared_inputs = prepare_inputs(path_imgs, file_format, n_channels)

        expected_inputs = path_imgs

        assert prepared_inputs == expected_inputs

    @pytest.mark.unit
    def test_prepare_inputs_grayscale_to_3channel_exits(self):
        path_imgs = [self.testPath / '__test_files__'/ '__test_demo_files__' / 'image.png']
        file_format = '.png'
        n_channels = 3

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            prepare_inputs(path_imgs, file_format, n_channels)

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 2)

    @pytest.mark.unit
    def test_prepare_inputs_rgb_to_3channel_valid(self):
        path_imgs = [self.testPath / '__test_files__'/ '__test_demo_files__' / 'image_axonmyelin_index.png']
        file_format = '.png'
        n_channels = 3
        prepared_inputs = prepare_inputs(path_imgs, file_format, n_channels)

        expected_inputs = path_imgs

        assert prepared_inputs == expected_inputs

    @pytest.mark.unit
    def test_prepare_inputs_rgb_to_grayscale(self):
        path_imgs = [self.testPath / '__test_files__'/ '__test_demo_files__' / 'image_axonmyelin_index.png']
        file_format = '.png'
        n_channels = 1
        prepared_inputs = prepare_inputs(path_imgs, file_format, n_channels)

        expected_inputs = path_imgs

        assert prepared_inputs != expected_inputs
        for file in prepared_inputs:
            assert Path(file).exists
            self.to_delete.append(Path(file))


    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.segment.main(["-i", str(self.imagePath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.segment.main(["-i", str(self.imageFolderPath), "-v", "1"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.integration
    def test_main_cli_creates_logfile(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.segment.main(["-i", str(self.imagePath), "-v", "1"])

        assert Path('ads_base.log').exists()

    @pytest.mark.integration
    def test_main_cli_fails_for_incorrect_file_extention(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            ads_base.segment.main(["-i", str(self.modelPath / 'dataset.json')])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 1)
