# coding: utf-8

from pathlib import Path
import random
import string
import pytest
import shutil
import glob
import sys

import AxonDeepSeg
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.morphometrics.launch_morphometrics_computation import launch_morphometrics_computation
from config import axonmyelin_suffix, axon_suffix, myelin_suffix, morph_suffix, index_suffix, axonmyelin_index_suffix


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent
        self.dataPath = self.testPath / '__test_files__' / '__test_demo_files__'

        self.morphometricsFile =  "image" + "_" + str(morph_suffix)
        self.axon_shape = "ellipse"         # axon shape is set to ellipse
        self.morphometricsPath = self.dataPath / self.morphometricsFile

    def teardown(self):

        if self.morphometricsPath.exists():
            self.morphometricsPath.unlink()

    # --------------launch_morphometrics_computation tests-------------- #
    @pytest.mark.unit
    def test_launch_morphometrics_computation_saves_expected_files(self):
        expectedFiles = {'aggregate_morphometrics.txt',
                         'AxonDeepSeg_map-axondiameter.png',
                         'axonlist.npy'
                         }

        pathImg = self.dataPath / 'image.png'
        pathPrediction = self.dataPath / ('image' + str(axonmyelin_suffix))

        launch_morphometrics_computation(str(pathImg), str(pathPrediction))

        for fileName in expectedFiles:
            fullFilePath = self.dataPath / fileName
            assert fullFilePath.is_file()
            fullFilePath.unlink()

    @pytest.mark.unit
    def test_launch_morphometrics_computation_saves_expected_files_with_axon_as_ellipse(self):
        expectedFiles = {'aggregate_morphometrics.txt',
                         'AxonDeepSeg_map-axondiameter.png',
                         'axonlist.npy'
                         }

        pathImg = self.dataPath / 'image.png'
        pathPrediction = self.dataPath / ('image' + str(axonmyelin_suffix))

        launch_morphometrics_computation(str(pathImg), str(pathPrediction), axon_shape=self.axon_shape)

        for fileName in expectedFiles:
            fullFilePath = self.dataPath / fileName
            assert fullFilePath.is_file()
            fullFilePath.unlink()

    @pytest.mark.unit
    def test_launch_morphometrics_computation_errors_for_missing_file(self):
        nonExistingFile = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        pathPrediction = self.dataPath / ('image' + str(axonmyelin_suffix))

        with pytest.raises((IOError, OSError)):
            launch_morphometrics_computation(str(nonExistingFile), str(pathPrediction))

    # --------------main (cli) tests-------------- #
    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs(self):
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-s", "0.07", "-i", str(pathImg), "-f", str(morph_suffix)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and self.morphometricsPath.exists()
    
    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input_with_pixel_size_file(self):
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and self.morphometricsPath.exists()

    def test_main_cli_runs_succesfully_with_valid_inputs_for_save_morphometrics_as_csv(self):
        pathImg = self.dataPath / 'image.png'

        self.morphometricsFile = pathImg.stem + "_" + morph_suffix.stem + ".csv"
        self.morphometricsPath = self.dataPath / self.morphometricsFile

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), '-f', (morph_suffix.stem + '.csv')])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and self.morphometricsPath.exists()

    @pytest.mark.unit
    def test_main_cli_successfully_outputs_index_and_colored_image(self):
        expected_outut_images_filenames = \
            [self.dataPath / ("image" + str(index_suffix)), self.dataPath / ("image" + str(axonmyelin_index_suffix))]
        pathImg = self.dataPath / 'image.png'

        self.morphometricsFile = "axon_morphometrics.csv"
        self.morphometricsPath = self.dataPath / self.morphometricsFile

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), '-f', 'axon_morphometrics.csv'])

        assert expected_outut_images_filenames[0].exists() and expected_outut_images_filenames[1].exists()

    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_for_custom_morphometrics_file_name(self):
        pathImg = self.dataPath / 'image.png'
        self.morphometricsFile = 'test_morphometrics.xlsx'
        self.morphometricsPath = self.dataPath / (pathImg.stem + '_' + self.morphometricsFile)

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), "-f", "test_morphometrics.xlsx"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and self.morphometricsPath.exists()

        # unlink the morphometrics file
        self.morphometricsPath.unlink()
    
    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_with_axon_shape_as_ellipse(self):
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), "-a", "ellipse"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and self.morphometricsPath.exists()
    
    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_with_axon_shape_as_circle(self):
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), "-a", "circle"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and self.morphometricsPath.exists()

    @pytest.mark.unit
    def test_main_cli_runs_successfully_for_generating_morphometrics_multiple_images(self):
        pathImg = self.dataPath / 'image.png'
        
        # path of `__test_demo_files__` directory
        pathDirCopy = self.dataPath.parent / '__test_demo_files_copy__'

        if not pathDirCopy.exists():
            # Make a copy of `__test_demo_files__` directory
            shutil.copytree(self.dataPath, pathDirCopy, copy_function=shutil.copy)
        pathImgCopy = self.dataPath.parent / '__test_demo_files_copy__' / 'image.png'
        morphometricsPathCopy = self.dataPath.parent / '__test_demo_files_copy__' / self.morphometricsFile

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), str(pathImgCopy)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and self.morphometricsPath.exists() and morphometricsPathCopy.exists()
        
        # unlink the morphometrics file
        morphometricsPathCopy.unlink() 

    @pytest.mark.unit 
    def test_main_cli_runs_successfully_for_generating_batches_morphometrics_multiple_images(self):
        
        # path of `__test_demo_files__` directory
        pathDirCopy = self.dataPath.parent / '__test_demo_files_copy__'

        if not pathDirCopy.exists():
            # Make a copy of `__test_demo_files__` directory
            shutil.copytree(self.dataPath, pathDirCopy, copy_function=shutil.copy)
        
        list_images = glob.glob(str(pathDirCopy / 'image*.png'))

        for image in list_images:
            img = image.replace("image", "img") 
            shutil.copy(pathDirCopy / Path(image), pathDirCopy / Path(img)) # duplicate the images to test batch morphometrics CLI command

        morphometricsImagePathCopy = self.dataPath.parent / '__test_demo_files_copy__' / self.morphometricsFile # morphometrics file of `image.png` image
        morphometricsImgPathCopy = self.dataPath.parent / '__test_demo_files_copy__' / ('img' + '_' + str(morph_suffix)) # morphometrics file of `img.png` image

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathDirCopy)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0) and morphometricsImagePathCopy.exists() and morphometricsImgPathCopy.exists()
        
        # unlink the morphometrics file
        morphometricsImagePathCopy.unlink() 
        morphometricsImgPathCopy.unlink() 

        # remove the duplicated images
        list_images = glob.glob(str(pathDirCopy / 'img*.png')) # list of duplicated images
        for image in list_images:
            if  Path(image).exists():
                Path(image).unlink()

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_if_image_is_not_segmented(self):
        self.dataPath = self.testPath / '__test_files__' / '__test_segment_files__'
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)
    
    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_for_myelin_mask_not_present(self):
        pathImg = self.dataPath / 'image.png'

        # myelin mask path
        pathMyelin = self.dataPath / ('image' + str(myelin_suffix))

        # Read the myelin mask
        myelinMask = ads.imread(str(pathMyelin))

        # Delete the myelin mask for exception
        if pathMyelin.exists():
            pathMyelin.unlink()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        # Save the myelin mask back to the `__test_demo_files__`
        ads.imwrite(str(pathMyelin), myelinMask)

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)

    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_for_axon_mask_not_present(self):
        pathImg = self.dataPath / 'image.png'
        
        # axon mask path
        pathAxon = self.dataPath / ('image' + str(axon_suffix))

        # Read the axon mask
        axonMask = ads.imread(str(pathAxon))

        if pathAxon.exists():
            pathAxon.unlink()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        # Save the axon mask back to the `__test_demo_files__`
        ads.imwrite(str(pathAxon), axonMask)

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)
