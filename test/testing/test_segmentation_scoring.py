# coding: utf-8

import pytest
import os
import numpy as np
from scipy.misc import imread
import pandas as pd
from AxonDeepSeg.testing.segmentation_scoring import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the test directory, "test/"
        self.testPath = os.path.split(self.fullPath)[0]

        self.folderPath = os.path.join(
            self.testPath,
            '__test_files__',
            '__test_demo_files__'
            )

        self.img = imread(
            os.path.join(self.folderPath, 'image.png'),
            flatten=True
            )

        self.groundtruth = imread(
            os.path.join(self.folderPath, 'mask.png'),
            flatten=True
            )

        self.prediction = imread(
            os.path.join(self.folderPath, 'AxonDeepSeg_seg-axonmyelin.png'),
            flatten=True)

    def teardown(self):
        pass

    # --------------score_analysis tests-------------- #
    @pytest.mark.unit
    def test_score_analysis_returns_nonzero_outputs_diff_masks(self):
        gtAxon = self.groundtruth > 200
        predAxon = self.prediction > 200

        [sensitivity, precision, diffusion] = score_analysis(self.img, gtAxon, predAxon)

        # Note that if imageio.imread(imagefile) was used instead of scipy's
        # imread with flatten = true, the scores would be 0 (from experience)
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
    def test_score_analysis_runs_succesfully_with_visualization_on(self):

        assert score_analysis(
            self.img,
            self.groundtruth,
            self.prediction,
            visualization=True
            )

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
