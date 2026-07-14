# coding: utf-8

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from AxonDeepSeg.apply_model import (
    get_checkpoint_name,
    extract_from_nnunet_prediction,
    find_folds,
    get_predictor,
    clear_predictor_cache,
    axon_segmentation,
    segment_image_array
)

from AxonDeepSeg import ads_utils
from AxonDeepSeg.params import nnunet_suffix, axon_suffix, myelin_suffix, axonmyelin_suffix

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


        self.temp_files = []
        clear_predictor_cache()

    def teardown_method(self):
        for files in self.temp_files:
            files.unlink()
        # A predictor left in the cache would leak into the next test.
        clear_predictor_cache()

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

    # --------------get_predictor tests-------------- #
    @pytest.mark.unit
    def test_get_predictor_loads_the_checkpoint_only_once(self):
        # The checkpoint is ~256 MB. Reloading it per call is what made a warm
        # container exactly as slow as a cold one.
        with patch('AxonDeepSeg.apply_model.nnUNetPredictor') as mock_predictor:
            get_predictor(self.nnunetModelLight, 'light', -1)
            get_predictor(self.nnunetModelLight, 'light', -1)

        instance = mock_predictor.return_value
        assert instance.initialize_from_trained_model_folder.call_count == 1

    @pytest.mark.unit
    def test_get_predictor_returns_the_same_instance(self):
        with patch('AxonDeepSeg.apply_model.nnUNetPredictor'):
            first = get_predictor(self.nnunetModelLight, 'light', -1)
            second = get_predictor(self.nnunetModelLight, 'light', -1)

        assert first is second

    @pytest.mark.unit
    def test_get_predictor_reloads_for_a_different_device(self):
        # The device is fixed at nnUNetPredictor construction, so a different
        # gpu_id must not reuse the CPU predictor.
        with patch('AxonDeepSeg.apply_model.nnUNetPredictor') as mock_predictor:
            get_predictor(self.nnunetModelLight, 'light', -1)
            get_predictor(self.nnunetModelLight, 'light', 0)

        assert mock_predictor.call_count == 2

    @pytest.mark.unit
    def test_clear_predictor_cache_forces_a_reload(self):
        with patch('AxonDeepSeg.apply_model.nnUNetPredictor') as mock_predictor:
            get_predictor(self.nnunetModelLight, 'light', -1)
            clear_predictor_cache()
            get_predictor(self.nnunetModelLight, 'light', -1)

        assert mock_predictor.call_count == 2

    # --------------segment_image_array tests-------------- #
    @pytest.mark.integration
    def test_segment_image_array_matches_the_file_based_path(self):
        # The in-memory path skips nnU-Net's per-call worker pool (~80% of a GPU
        # request) by calling predict_single_npy_array instead of predict_from_files.
        # It must produce exactly what the file-based path produces.
        demo_image = (
            self.projectPath /
            'test' /
            '__test_files__' /
            '__test_demo_files__' /
            'image.png'
        )

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            staged = tmp_dir / 'input.png'
            shutil.copy(demo_image, staged)

            axon_segmentation(
                path_inputs=[staged],
                path_model=self.nnunetModelLight,
                model_type='light',
            )
            from_files = {
                'axon': ads_utils.imread(tmp_dir / ('input' + str(axon_suffix))),
                'myelin': ads_utils.imread(tmp_dir / ('input' + str(myelin_suffix))),
                'axonmyelin': ads_utils.imread(tmp_dir / ('input' + str(axonmyelin_suffix))),
            }

            in_memory = segment_image_array(
                ads_utils.imread(demo_image),
                self.nnunetModelLight,
                model_type='light',
            )
        finally:
            shutil.rmtree(tmp_dir)

        for mask_name, expected in from_files.items():
            assert np.array_equal(in_memory[mask_name], expected), mask_name
