# coding: utf-8

from pathlib import Path
import pytest

from ads_base.apply_model import (
    get_checkpoint_name, 
    extract_from_nnunet_prediction,
    find_folds
)

from ads_base import ads_utils
from ads_base.params import nnunet_suffix

import numpy as np

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.testPath = Path(__file__).resolve().parent
        self.projectPath = self.testPath.parent

        self.checkpointFolder = (
            self.projectPath /
            'test' /
            '__test_files__' /
            '__test_checkpoint_files__'
            )

        self.nnunetFolder = (
            self.projectPath /
            'test' /
            '__test_files__' /
            '__test_nnunet_files__'
            )
        
        self.nnunetFile = (
            self.nnunetFolder /
            'image_seg-nnunet.png'
        )

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


        self.temp_files = []

    def teardown_method(self):
        for files in self.temp_files:
            files.unlink()

    # --------------get_checkpoint_name tests-------------- #
    @pytest.mark.unit
    def test_get_checkpoint_name_case1(self):
        assert get_checkpoint_name(self.checkpointFolder / "case1") == 'checkpoint_best.pth'
       
    @pytest.mark.unit
    def test_get_checkpoint_name_case2(self):
        assert get_checkpoint_name(self.checkpointFolder / "case2") == 'checkpoint_final.pth'

    @pytest.mark.unit
    def test_get_checkpoint_name_case3(self):
        assert get_checkpoint_name(self.checkpointFolder / "case3") == 'checkpoint_2.pth'

    # --------------extract_from_nnunet_prediction tests-------------- #
    @pytest.mark.unit
    def test_extract_from_nnunet_prediction_does_not_throws_value_error_for_class(self):
        pred_path = self.nnunetFile
        pred = ads_utils.imread(pred_path)
        class_name = 'TestClass'
        class_value = 123 # Not a class (pixel value) in the nnunet file (image)
        try:
            extract_from_nnunet_prediction(pred, pred_path, class_name, class_value)
        except ValueError:
            pytest.fail('Case must not throw error, only warning')
        else:
            pass

    @pytest.mark.unit
    def test_extract_from_nnunet_prediction_throws_name_error_for_nnunet_file(self):
        pred_path = 'filename.png' # doesn't have the suffix, check next:
        assert str(nnunet_suffix) not in str(pred_path)

        pred = np.ones(1)
        class_name = 'TestClass'
        class_value = 1 
        try:
            extract_from_nnunet_prediction(pred, pred_path, class_name, class_value)
        except NameError:
            pass
        else:
            pytest.fail('Excepted filename not to have ' + nnunet_suffix + ' in filename')

    @pytest.mark.unit
    def test_extract_from_nnunet_prediction_returns_expected_filename(self):
        pred_path = self.nnunetFile
        pred = ads_utils.imread(pred_path)
        class_name = 'axon'
        class_value = 2
        
        output_filename = extract_from_nnunet_prediction(pred, pred_path, class_name, class_value)
        self.temp_files.append(Path(output_filename))

        expected_filename = 'image_seg-axon.png'
        assert Path(output_filename).name == expected_filename

    # --------------extract_from_nnunet_prediction tests-------------- #
    @pytest.mark.unit
    def test_find_folds_light(self):
        path_model = self.nnunetModelLight
        model_type = 'light'

        folds_avail = find_folds(path_model, model_type)

        expected_folds_avail = ['all']

        assert folds_avail == expected_folds_avail

    @pytest.mark.unit
    def test_find_folds_else_light(self):
        path_model = self.nnunetModelLight
        model_type = 'fake_light' # Just use the light model folder 
                                  # already downloaded to probe the else case, 
                                  # should still give ['all']

        folds_avail = find_folds(path_model, model_type)

        expected_folds_avail = ['all']

        assert folds_avail == expected_folds_avail


    @pytest.mark.unit
    def test_find_folds_else_ensemble(self):
        path_model = self.nnunetModelEmptyEnsemble
        model_type = 'ensemble'

        folds_avail = find_folds(path_model, model_type)
        folds_avail.sort()

        expected_folds_avail = ['0', '1', '2', '3', '4']

        assert folds_avail == expected_folds_avail
