# coding: utf-8

from pathlib import Path
import random
import string
import pytest

from AxonDeepSeg.morphometrics.launch_morphometrics_computation import *
from AxonDeepSeg.segment import *
from config import axonmyelin_suffix, myelin_suffix, axon_suffix


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent
        self.dataPath = self.testPath / '__test_files__' / '__test_demo_files__'

        self.axon_shape = "ellipse"         # axon shape is set to ellipse
        self.morphometricsFile = "Morphometrics.xlsx"

    def teardown(self):
        self.morphometricsPath = self.dataPath / self.morphometricsFile

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
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-s", "0.07", "-i", str(pathImg), "-f", "Morphometrics", "-a", "circle"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)
    
    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_for_folder_input_with_pixel_size_file(self):
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_with_axon_shape_as_ellipse(self):
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), "-a", "ellipse"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)
    
    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_with_axon_shape_as_circle(self):
        pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), "-a", "circle"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)

    @pytest.mark.unit
    def test_main_cli_runs_succesfully_with_valid_inputs_for_custom_morphometrics_file_name(self):
        pathImg = self.dataPath / 'image.png'
        self.morphometricsFile = "test_morphometrics.xlsx"

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg), "-f", str(self.morphometricsFile)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 0)
    
    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_if_image_is_not_segmented(self):
        self.dataPath = self.testPath / '__test_files__' / '__test_segment_files__'
        pathImg = pathImg = self.dataPath / 'image.png'

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)
    
    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_for_pixel_size_not_provided(self):
        self.dataPath = self.testPath / '__test_files__' / '__test_segment_files__'
        pathImg = pathImg = self.dataPath / 'image.png'

        
        #Segment `image.png` present in `__test_segment_files__` directory
        AxonDeepSeg.segment.main(["-t", "SEM", "-i", str(pathImg), "-v", "2", "-s", "0.37"])

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)

    '''
    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_for_myelin_mask_not_present(self):
        pathImg = pathImg = self.dataPath / 'image.png'
        
        #myelin mask path
        pathMyelin = self.dataPath / ('image' + str(myelin_suffix))

        if pathMyelin.exists():
            pathMyelin.unlink()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)
    
    @pytest.mark.exceptionhandling
    def test_main_cli_handles_exception_for_axon_mask_not_present(self):
        pathImg = pathImg = self.dataPath / 'image.png'
        
        #axon mask path
        pathAxon = self.dataPath / ('image' + str(axon_suffix))

        if pathAxon.exists():
            pathAxon.unlink()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.morphometrics.launch_morphometrics_computation.main(["-i", str(pathImg)])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == 3)
    '''
