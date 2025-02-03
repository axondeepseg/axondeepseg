# coding: utf-8

import os
from pathlib import Path
import numpy as np
import imageio
import pandas as pd

import pytest

from ads_base.testing.segmentation_scoring import score_analysis, dice, pw_dice, Metrics_calculator
from ads_base import ads_utils
from ads_base.params import axonmyelin_suffix


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.folderPath = self.testPath / '__test_files__' / '__test_demo_files__'

        self.img = ads_utils.imread(self.folderPath / 'image.png')

        self.groundtruth = ads_utils.imread(self.folderPath / 'mask.png')

        self.prediction = ads_utils.imread(
            self.folderPath / ('image' + str(axonmyelin_suffix))
            )

    def teardown_method(self):
        pass

    # --------------score_analysis tests-------------- #
    @pytest.mark.unit
    def test_score_analysis_returns_nonzero_outputs_diff_masks(self):
        gtAxon = self.groundtruth > 200
        predAxon = self.prediction > 200

        [sensitivity, precision, diffusion] = score_analysis(self.img, gtAxon, predAxon)

        # Note that if imageio.v2.imread(imagefile) was used instead of scipy's
        # imread with as_gray = true, the scores would be 0 (from experience)
        assert sensitivity != 0.0
        assert precision != 0.0
        assert diffusion != 0.0

    @pytest.mark.unit
    def test_score_analysis_returns_expected_outputs_for_same_masks(self):
        gtAxon = self.groundtruth > 200

        [sensitivity, precision, diffusion] = score_analysis(self.img, gtAxon, gtAxon)

        assert sensitivity == 1.0
        assert precision == 1.0
        assert diffusion == 0.0

    @pytest.mark.unit
    def test_score_analysis_runs_successfully_with_visualization_on(self):
        saved_dir = Path.cwd()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            # Test function
            assert score_analysis(
                self.img,
                self.groundtruth,
                self.prediction,
                visualization=True
                )
            # Test success of visualisation
            pred_path = Path(tmpdir) / "prediction.png"
            gt_path = Path(tmpdir) / "ground_truth.png"
            assert pred_path.is_file()
            assert gt_path.is_file()

            # Go back to previous dir
            os.chdir(saved_dir)

    # --------------dice tests-------------- #
    @pytest.mark.unit
    def test_dice_returns_nonempty_pandas_dataframe(self):

        gtAxon = self.groundtruth > 200
        predAxon = self.prediction > 200
        diceVals = dice(self.img, gtAxon, predAxon)

        assert not diceVals.empty

    # --------------pw_dice tests-------------- #
    @pytest.mark.unit
    def test_pwdice_returns_nonzero_value_for_diff_masks(self):

        gtMyelin = np.logical_and(
            self.groundtruth >= 50,
            self.groundtruth <= 200
            )

        predMyelin = np.logical_and(
            self.prediction >= 50,
            self.prediction <= 200
            )

        pwDiceVal = pw_dice(gtMyelin, predMyelin)

        assert pwDiceVal != 0

    @pytest.mark.unit
    def test_pwdice_returns_1_for_identical_masks(self):

        gtMyelin = np.logical_and(
            self.groundtruth >= 50,
            self.groundtruth <= 200
            )

        pwDiceVal = pw_dice(gtMyelin, gtMyelin)

        assert pwDiceVal == 1.0

    # --------------Metrics_calculator class tests-------------- #
    @pytest.mark.unit
    def test_Metrics_calculator_class_return_nonzeros_for_diff_masks(self):

        gtMyelin = np.logical_and(
            self.groundtruth >= 50,
            self.groundtruth <= 200
            )

        predMyelin = np.logical_and(
            self.prediction >= 50,
            self.prediction <= 200
            )

        axonMetrics = Metrics_calculator(gtMyelin, predMyelin)

        assert axonMetrics.pw_sensitivity() != 0
        assert axonMetrics.pw_precision() != 0
        assert axonMetrics.pw_specificity() != 0
        assert axonMetrics.pw_FN_rate() != 0
        assert axonMetrics.pw_FP_rate() != 0
        assert axonMetrics.pw_accuracy() != 0
        assert axonMetrics.pw_F1_score() != 0
        assert axonMetrics.pw_dice() != 0
        assert axonMetrics.pw_jaccard() != 0

        ewDict = axonMetrics.ew_dice()
        for key in ewDict:
            assert ewDict[key] != 0

        assert axonMetrics.pw_hausdorff_distance() != 0

    @pytest.mark.unit
    def test_Metrics_calculator_class_same_masks_returns_expected_values(self):

        gtMyelin = np.logical_and(
            self.groundtruth >= 50,
            self.groundtruth <= 200
            )

        axonMetrics = Metrics_calculator(gtMyelin, gtMyelin)

        assert axonMetrics.pw_sensitivity() == 1.0
        assert axonMetrics.pw_precision() == 1.0
        assert axonMetrics.pw_specificity() == 1.0
        assert axonMetrics.pw_FN_rate() == 0.0
        assert axonMetrics.pw_FP_rate() == 0.0
        assert axonMetrics.pw_accuracy() == 1.0
        assert axonMetrics.pw_F1_score() == 1.0
        assert axonMetrics.pw_dice() == 1.0
        assert axonMetrics.pw_jaccard() == 1.0

        ewDict = axonMetrics.ew_dice()
        for key in ewDict:
            if key == 'std':
                assert ewDict[key] == 0.0
            else:
                assert ewDict[key] == 1.0

        assert axonMetrics.pw_hausdorff_distance() == 0.0
