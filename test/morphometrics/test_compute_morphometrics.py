# coding: utf-8

from pathlib import Path
import string
import random
import math
import shutil
import numpy as np
import pandas as pd
from AxonDeepSeg import ads_utils as ads
from AxonDeepSeg import params
import pytest

# AxonDeepSeg imports
from AxonDeepSeg.visualization.simulate_axons import SimulateAxons, calc_myelin_thickness
from AxonDeepSeg.visualization.get_masks import get_masks
from AxonDeepSeg.morphometrics.compute_morphometrics import (
                                                                get_pixelsize,
                                                                get_axon_morphometrics,
                                                                rearrange_column_names_for_saving,
                                                                rename_column_names_after_loading,
                                                                save_axon_morphometrics,
                                                                load_axon_morphometrics,
                                                                draw_axon_diameter,
                                                                get_aggregate_morphometrics,
                                                                write_aggregate_morphometrics
                                                            )
from config import axonmyelin_suffix, axon_suffix, myelin_suffix


class TestCore(object):
    def setup(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        self.test_folder_path = (
            self.testPath /
            '__test_files__' /
            '__test_demo_files__'
            )
        self.pixelsizeValue = 0.07   # For current demo data.
        self.axon_shape = "ellipse"   # axon shape is set to ellipse

        pred_axon_path = self.test_folder_path / ('image' + str(axon_suffix))
        self.pred_axon = ads.imread(pred_axon_path)
        pred_myelin_path = self.test_folder_path / ('image' + str(myelin_suffix))
        self.pred_myelin = ads.imread(pred_myelin_path)

        # Image to test NaN output
        bad_pred_axon_path = self.test_folder_path / ('invalid_gratio' + str(axon_suffix))
        self.bad_pred_axon = ads.imread(bad_pred_axon_path)
        bad_pred_myelin_path = self.test_folder_path / ('invalid_gratio' + str(myelin_suffix))
        self.bad_pred_myelin = ads.imread(bad_pred_myelin_path)

        # simulated image of axon myelin where axon shape is circle; `plane angle = 0`
        self.path_sim_image_circle = (
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons_shape_circle.png'
        )

        self.image_sim = SimulateAxons()

        self.image_sim.generate_axon(axon_radius=50, center=[100, 100], gratio=0.9, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=45, center=[200, 200], gratio=0.8, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=40, center=[300, 300], gratio=0.7, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=35, center=[400, 400], gratio=0.6, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=30, center=[520, 520], gratio=0.5, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=23, center=[630, 630], gratio=0.4, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=18, center=[725, 725], gratio=0.3, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=12, center=[830, 830], gratio=0.2, plane_angle=0)
        self.image_sim.generate_axon(axon_radius=6, center=[920, 920], gratio=0.1, plane_angle=0)

        self.image_sim.save(self.path_sim_image_circle)

        self.tmpDir = self.fullPath / '__tmp__'
        if not self.tmpDir.exists():
            self.tmpDir.mkdir()

        self.image_sim_ellipse_path = Path(
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons_ellipse.png'
        )

        # Simulated image of axon and myelin to test morphometrics generated by ADS when axon shape is ellipse
        self.image_sim_ellipse = SimulateAxons()
        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[100, 100], gratio=0.9, plane_angle=0)
        self.image_sim_ellipse.generate_axon(axon_radius=45, center=[200, 200], gratio=0.8, plane_angle=10)
        self.image_sim_ellipse.generate_axon(axon_radius=40, center=[300, 300], gratio=0.7, plane_angle=20)
        self.image_sim_ellipse.generate_axon(axon_radius=35, center=[400, 400], gratio=0.6, plane_angle=30)
        self.image_sim_ellipse.generate_axon(axon_radius=30, center=[520, 520], gratio=0.5, plane_angle=40)
        self.image_sim_ellipse.generate_axon(axon_radius=23, center=[630, 630], gratio=0.4, plane_angle=50)
        self.image_sim_ellipse.generate_axon(axon_radius=18, center=[720, 750], gratio=0.3, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=12, center=[800, 900], gratio=0.2, plane_angle=70)

        self.image_sim_ellipse.save(self.image_sim_ellipse_path)

        # simulated axons to test --border-info option
        self.image_sim_border_info_path = Path(
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'image_border_touching_simulation.png'
        )

        simulated = SimulateAxons()
        # axon touching image border
        x_pos = 100
        y_pos = 100
        axon_radius = 100
        gratio = 0.6
        myelin_thickness = calc_myelin_thickness(axon_radius, gratio)
        simulated.generate_axon(axon_radius=axon_radius, center=[x_pos, y_pos], gratio=gratio, plane_angle=0)

        # axon not touching any image border
        x_pos = 500
        y_pos = 500
        axon_radius = 40
        gratio = 0.6
        myelin_thickness = calc_myelin_thickness(axon_radius, gratio)
        simulated.generate_axon(axon_radius=axon_radius, center=[x_pos, y_pos], gratio=gratio, plane_angle=0)

        simulated.save(self.image_sim_border_info_path)


    def teardown(self):
        if self.tmpDir.exists():
            shutil.rmtree(self.tmpDir)
        if self.image_sim_ellipse_path.is_file():
            self.image_sim_ellipse_path.unlink()
        if self.image_sim_border_info_path.exists():
            self.image_sim_border_info_path.unlink()


    # --------------get_pixelsize tests-------------- #
    @pytest.mark.unit
    def test_get_pixelsize_returns_expected_value(self):
        expectedValue = self.pixelsizeValue
        pixelsizeFileName = self.test_folder_path / 'pixel_size_in_micrometer.txt'
        actualValue = get_pixelsize(str(pixelsizeFileName))

        assert actualValue == expectedValue

    @pytest.mark.unit
    def test_get_pixelsize_throws_error_for_nonexisisting_file(self):
        nonExistingFile = ''.join(
            random.choice(string.ascii_lowercase) for i in range(16))

        with pytest.raises(IOError):
            get_pixelsize(str(nonExistingFile))

    @pytest.mark.unit
    def test_get_pixelsize_throws_error_for_invalid_data_file(self):
        tmpName = self.tmpDir / 'tmpInvalid.txt'
        with open(tmpName, 'wb') as tmp:

            tmp.write('&&&'.encode())

        with pytest.raises(ValueError):

            get_pixelsize(str(tmpName))

    # --------------get_axon_morphometrics tests-------------- #
    @pytest.mark.unit
    def test_get_axon_morphometrics_returns_expected_type(self):
        stats_dataframe = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))
        assert isinstance(stats_dataframe, pd.DataFrame)

    @pytest.mark.unit
    def test_get_axon_morphometrics_returns_expected_type_with_axon_as_ellipse(self):
        stats_dataframe = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path), axon_shape=self.axon_shape)
        assert isinstance(stats_dataframe, pd.DataFrame)

    @pytest.mark.unit
    def test_get_axon_morphometrics_returns_expected_columns(self):
        expected_columns = {'y0',
                            'x0',
                            'axon_diam',
                            'axon_area',
                            'axon_perimeter',
                            'axonmyelin_perimeter',
                            'solidity',
                            'eccentricity',
                            'orientation'
                           }

        stats_dataframe = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))

        for column in stats_dataframe.columns:
            assert column in expected_columns

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask(self):
        stats_dataframe = get_axon_morphometrics(
            self.pred_axon,
            str(self.test_folder_path),
            im_myelin=self.pred_myelin
            )
        print("The values are ", stats_dataframe['gratio'][1], stats_dataframe['axon_diam'][1], stats_dataframe['myelin_thickness'][1])
        assert stats_dataframe['gratio'][1] == pytest.approx(0.74, rel=0.01)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask_with_axon_as_ellipse(self):
        stats_dataframe = get_axon_morphometrics(
            self.pred_axon,
            str(self.test_folder_path),
            im_myelin=self.pred_myelin,
            axon_shape=self.axon_shape
            )
        print("The values are ", stats_dataframe['gratio'][1], stats_dataframe['axon_diam'][1], stats_dataframe['myelin_thickness'][1])

        assert stats_dataframe['gratio'][1] == pytest.approx(0.78, rel=0.01)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_invalid_gratio_with_axon_as_ellipse(self):
        stats_dataframe = get_axon_morphometrics(
            self.bad_pred_axon,
            str(self.test_folder_path),
            im_myelin=self.bad_pred_myelin,
            axon_shape=self.axon_shape
            )
        assert np.isnan(stats_dataframe['gratio'][0])

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask_simulated_axons(self):
        path_pred = (
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons_shape_circle.png'
        )

        gratio_sim = np.array([
                                0.9,
                                0.8,
                                0.7,
                                0.6,
                                0.5,
                                0.4,
                                0.3,
                                0.2,
                                0.1
                                ])

        axon_diam_sim = np.array([
                                100,
                                90,
                                80,
                                70,
                                60,
                                46,
                                36,
                                24,
                                12
                                ])

        myelin_thickness_sim = (axon_diam_sim / 2) * (1/gratio_sim - 1)

        # read paths and compute axon/myelin masks
        pred = ads.imread(path_pred)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        # compute axon morphometrics
        stats_dataframe = get_axon_morphometrics(pred_axon, str(path_pred.parent), im_myelin=pred_myelin)

        for ii in range(0, len(gratio_sim)):
            assert stats_dataframe['gratio'][ii] == pytest.approx(gratio_sim[ii], rel=0.1)
            assert stats_dataframe['axon_diam'][ii] == pytest.approx(axon_diam_sim[ii], rel=0.1)
            assert stats_dataframe['myelin_thickness'][ii] == pytest.approx(myelin_thickness_sim[ii], rel=0.1)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask_simulated_axons_with_axon_as_ellipse_all_argument_varies(self):
        # simulated image of axon and myelin to test morphoterics generated by ADS when axon shape is ellipse

        # varying all the arguments: gratio, plane_angle, axon_radius
        self.image_sim_ellipse = SimulateAxons()

        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[80, 80], gratio=0.9, plane_angle=0)
        self.image_sim_ellipse.generate_axon(axon_radius=45, center=[180, 180], gratio=0.8, plane_angle=10)
        self.image_sim_ellipse.generate_axon(axon_radius=40, center=[280, 280], gratio=0.7, plane_angle=20)
        self.image_sim_ellipse.generate_axon(axon_radius=35, center=[380, 380], gratio=0.6, plane_angle=30)
        self.image_sim_ellipse.generate_axon(axon_radius=30, center=[480, 480], gratio=0.5, plane_angle=40)
        self.image_sim_ellipse.generate_axon(axon_radius=23, center=[580, 580], gratio=0.4, plane_angle=50)
        self.image_sim_ellipse.generate_axon(axon_radius=18, center=[700, 700], gratio=0.3, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=12, center=[800, 850], gratio=0.2, plane_angle=70)

        self.image_sim_ellipse.save(self.image_sim_ellipse_path)

        gratio_sim = np.array([
                                0.9,
                                0.8,
                                0.7,
                                0.6,
                                0.5,
                                0.4,
                                0.3,
                                0.2,
                                ])

        axon_diam_sim = np.array([
                                100,
                                90,
                                80,
                                70,
                                60,
                                46,
                                36,
                                24,
                                ])

        myelin_thickness_sim = (axon_diam_sim / 2) * ((1/gratio_sim) - 1)

        # read paths and compute axon/myelin masks
        pred = ads.imread(self.image_sim_ellipse_path)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        # compute axon morphometrics
        stats_dataframe = get_axon_morphometrics(pred_axon, str(self.image_sim_ellipse_path.parent), im_myelin=pred_myelin, axon_shape=self.axon_shape)

        for ii in range(0, len(gratio_sim)):
            assert stats_dataframe['gratio'][ii] == pytest.approx(gratio_sim[ii], rel=0.1)
            assert stats_dataframe['axon_diam'][ii] == pytest.approx(axon_diam_sim[ii], rel=0.1)
            assert stats_dataframe['myelin_thickness'][ii] == pytest.approx(myelin_thickness_sim[ii], rel=0.1)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask_simulated_axons_with_axon_as_ellipse_gratio_varies(self):

        # varying g ratio
        self.image_sim_ellipse = SimulateAxons()


        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[115, 90], gratio=0.9, plane_angle=20)
        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[220, 220], gratio=0.6, plane_angle=20)
        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[360, 360], gratio=0.5, plane_angle=20)
        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[550, 550], gratio=0.4, plane_angle=20)
        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[800, 800], gratio=0.3, plane_angle=20)

        self.image_sim_ellipse.save(self.image_sim_ellipse_path)

        gratio_sim = np.array([
                                0.9,
                                0.6,
                                0.5,
                                0.4,
                                0.3,
                                ])

        axon_diam_sim = np.array([
                                100,
                                100,
                                100,
                                100,
                                100,
                                ])

        myelin_thickness_sim = (axon_diam_sim / 2) * ((1/gratio_sim) - 1)

        # Read paths and compute axon/myelin masks
        pred = ads.imread(self.image_sim_ellipse_path)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        # compute axon morphometrics
        stats_dataframe = get_axon_morphometrics(pred_axon, str(self.image_sim_ellipse_path.parent), im_myelin=pred_myelin, axon_shape=self.axon_shape)

        for ii in range(0, len(gratio_sim)):

            assert stats_dataframe['gratio'][ii] == pytest.approx(gratio_sim[ii], rel=0.1)
            assert stats_dataframe['axon_diam'][ii] == pytest.approx(axon_diam_sim[ii], rel=0.1)
            assert stats_dataframe['myelin_thickness'][ii] == pytest.approx(myelin_thickness_sim[ii], rel=0.1)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask_simulated_axons_with_axon_as_ellipse_plane_angle_varies(self):

        # varying plane angle
        self.image_sim_ellipse = SimulateAxons()

        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[100, 100], gratio=0.5, plane_angle=0)
        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[200, 200], gratio=0.5, plane_angle=10)
        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[300, 300], gratio=0.5, plane_angle=20)
        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[400, 400], gratio=0.5, plane_angle=30)
        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[500, 500], gratio=0.5, plane_angle=40)
        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[600, 600], gratio=0.5, plane_angle=50)
        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[700, 700], gratio=0.5, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=25, center=[800, 800], gratio=0.5, plane_angle=70)

        self.image_sim_ellipse.save(self.image_sim_ellipse_path)

        gratio_sim = np.array([
                                0.5,
                                0.5,
                                0.5,
                                0.5,
                                0.5,
                                0.5,
                                0.5,
                                0.5,
                                ])

        axon_diam_sim = np.array([
                                50,
                                50,
                                50,
                                50,
                                50,
                                50,
                                50,
                                50
                                ])

        myelin_thickness_sim = (axon_diam_sim / 2) * (1/gratio_sim - 1)

        # read paths and compute axon/myelin masks
        pred = ads.imread(self.image_sim_ellipse_path)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        # compute axon morphometrics
        stats_dataframe = get_axon_morphometrics(pred_axon, str(self.image_sim_ellipse_path.parent), im_myelin=pred_myelin, axon_shape=self.axon_shape)

        for ii in range(0, len(gratio_sim)):
            assert stats_dataframe['gratio'][ii] == pytest.approx(gratio_sim[ii], rel=0.1)
            assert stats_dataframe['axon_diam'][ii] == pytest.approx(axon_diam_sim[ii], rel=0.1)
            assert stats_dataframe['myelin_thickness'][ii] == pytest.approx(myelin_thickness_sim[ii], rel=0.1)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask_simulated_axons_with_axon_as_ellipse_radius_varies(self):

        # varying axon radius
        self.image_sim_ellipse = SimulateAxons()

        self.image_sim_ellipse.generate_axon(axon_radius=50, center=[150, 100], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=45, center=[250, 250], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=40, center=[380, 380], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=35, center=[480, 480], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=30, center=[600, 600], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=23, center=[680, 680], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=18, center=[750, 750], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=12, center=[800, 800], gratio=0.7, plane_angle=60)
        self.image_sim_ellipse.generate_axon(axon_radius=6, center=[850, 850], gratio=0.7, plane_angle=60)

        self.image_sim_ellipse.save(self.image_sim_ellipse_path)

        gratio_sim = np.array([
                                0.7,
                                0.7,
                                0.7,
                                0.7,
                                0.7,
                                0.7,
                                0.7,
                                0.7,
                                0.7,
                                ])

        axon_diam_sim = np.array([
                                100,
                                90,
                                80,
                                70,
                                60,
                                46,
                                36,
                                24,
                                12,
                                ])

        myelin_thickness_sim = (axon_diam_sim / 2) * (1/gratio_sim - 1)

        # Read paths and compute axon/myelin masks
        pred = ads.imread(self.image_sim_ellipse_path)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        # Compute axon morphometrics
        stats_dataframe = get_axon_morphometrics(pred_axon, str(self.image_sim_ellipse_path.parent), im_myelin=pred_myelin, axon_shape=self.axon_shape)

        for ii in range(0, len(gratio_sim)):
            assert stats_dataframe['gratio'][ii] == pytest.approx(gratio_sim[ii], rel=0.1)
            assert stats_dataframe['axon_diam'][ii] == pytest.approx(axon_diam_sim[ii], rel=0.1)
            assert stats_dataframe['myelin_thickness'][ii] == pytest.approx(myelin_thickness_sim[ii], rel=0.1)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_unexpected_myelin_mask_simulated_axons(self):
        path_pred = (
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons_shape_circle.png'
            )

        pred = ads.imread(path_pred)
        pred_axon = pred > 200
        unexpected_pred_myelin = np.zeros(pred.shape)

        stats_dataframe = get_axon_morphometrics(
            pred_axon,
            str(path_pred.parent),
            im_myelin=unexpected_pred_myelin
            )

        for axon_index, axon_prop in stats_dataframe.iterrows():
            assert axon_prop['myelin_thickness'] == pytest.approx(0.0, rel=0.01)
            assert axon_prop['myelin_area'] == pytest.approx(0.0, rel=0.01)
            assert axon_prop['gratio'] == pytest.approx(1.0, rel=0.01)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_unexpected_myelin_mask_simulated_axons_with_axon_as_ellipse(self):
        pred = ads.imread(self.image_sim_ellipse_path)
        pred_axon = pred > 200
        unexpected_pred_myelin = np.zeros(pred.shape)

        stats_dataframe = get_axon_morphometrics(
            pred_axon,
            str(self.image_sim_ellipse_path.parent),
            im_myelin=unexpected_pred_myelin,
            axon_shape=self.axon_shape
            )

        for axon_index, axon_prop in stats_dataframe.iterrows():
            assert axon_prop['myelin_thickness'] == pytest.approx(0.0, rel=0.01)
            assert axon_prop['myelin_area'] == pytest.approx(0.0, rel=0.01)
            assert axon_prop['gratio'] == pytest.approx(1.0, rel=0.01)

    # --------------axon and myelin perimeter tests-------------- #
    @pytest.mark.unit
    def test_axon_perimeter_morphometrics_when_axon_shape_circle(self):

        # Simulated image path
        image_sim_path = Path(
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons_shape_circle.png'
        )

        axon_diam_sim = np.array([
                                    100,
                                    90,
                                    80,
                                    70,
                                    60,
                                    46,
                                    36,
                                    24,
                                    12
                                ])

        # axon perimeter (inner perimeter of myelin) = pi * diameter of axon
        axon_perimeter_sim = math.pi * axon_diam_sim

        pred = ads.imread(image_sim_path)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        stats_dataframe = get_axon_morphometrics(
            pred_axon,
            str(image_sim_path.parent),
            im_myelin=pred_myelin
            )

        for ii in range(0, len(axon_diam_sim)):
            assert stats_dataframe['axon_perimeter'][ii] == pytest.approx(axon_perimeter_sim[ii], rel=0.1)

        if image_sim_path.exists():
            image_sim_path.unlink()

    @pytest.mark.unit
    def test_axonmyelin_perimeter_morphometrics_when_axon_shape_circle(self):

        # Simulated image path
        image_sim_path = Path(
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons_circle_test_perimeter.png'
        )

        image_sim = SimulateAxons()

        image_sim.generate_axon(axon_radius=50, center=[100, 100], gratio=0.9, plane_angle=0)
        image_sim.generate_axon(axon_radius=45, center=[200, 200], gratio=0.8, plane_angle=0)
        image_sim.generate_axon(axon_radius=40, center=[300, 300], gratio=0.7, plane_angle=0)
        image_sim.generate_axon(axon_radius=35, center=[400, 400], gratio=0.6, plane_angle=0)
        image_sim.generate_axon(axon_radius=30, center=[520, 520], gratio=0.5, plane_angle=0)
        image_sim.generate_axon(axon_radius=23, center=[640, 640], gratio=0.4, plane_angle=0)
        image_sim.generate_axon(axon_radius=18, center=[725, 725], gratio=0.3, plane_angle=0)
        image_sim.generate_axon(axon_radius=12, center=[830, 830], gratio=0.2, plane_angle=0)
        image_sim.generate_axon(axon_radius=6, center=[920, 920], gratio=0.1, plane_angle=0)

        image_sim.save(image_sim_path)


        axon_diam_sim = np.array([
                                    100,
                                    90,
                                    80,
                                    70,
                                    60,
                                    46,
                                    36,
                                    24,
                                    12
                                ])

        gratio_sim = np.array([
                                0.9,
                                0.8,
                                0.7,
                                0.6,
                                0.5,
                                0.4,
                                0.3,
                                0.2,
                                0.1
                            ])

        # myelin thickness = radius_axon * ((1 / gratio) - 1 )
        myelin_thickness = (axon_diam_sim/2) * ((1 / gratio_sim) - 1)

        # axonmyelin perimeter (outer perimeter of myelin) = pi * diameter of axon + myelin
        axonmyelin_perimeter_sim = math.pi * ((myelin_thickness * 2) + axon_diam_sim)

        pred = ads.imread(image_sim_path)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        stats_dataframe = get_axon_morphometrics(
            pred_axon,
            str(image_sim_path.parent),
            im_myelin=pred_myelin
            )

        for ii in range(0, len(gratio_sim)):
            assert stats_dataframe['axonmyelin_perimeter'][ii] == pytest.approx(axonmyelin_perimeter_sim[ii], rel=0.1)

        if image_sim_path.exists():
            image_sim_path.unlink()

    # --------------save and load _axon_morphometrics tests-------------- #
    @pytest.mark.unit
    def test_save_axon_morphometrics_creates_file_in_expected_location(self):
        stats_dataframe = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))
        save_axon_morphometrics(str(self.tmpDir / 'morphometrics.pkl'), stats_dataframe)
        expectedFilePath = self.tmpDir / 'morphometrics.pkl'
        assert expectedFilePath.is_file()

    @pytest.mark.unit
    def test_save_axon_morphometrics_throws_error_if_folder_doesnt_exist(self):
        stats_dataframe = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))

        nonExistingFolder = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        nonExistingFolder = Path(nonExistingFolder)

        with pytest.raises(IOError):
            save_axon_morphometrics(str(nonExistingFolder / "morphometrics.pkl"), stats_dataframe)

    @pytest.mark.unit
    def test_load_axon_morphometrics_returns_identical_var_as_was_saved(self):
        original_stats_dataframe = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path), im_myelin=self.pred_myelin)
        morphometrics_file_path = self.tmpDir / "morphometrics.pkl"

        save_axon_morphometrics(morphometrics_file_path, original_stats_dataframe)

        # Load method only takes in a file path as an argument and returns the loaded dataframe
        loaded_stats_dataframe = load_axon_morphometrics(morphometrics_file_path)

        # Reorder the loaded dataframe so that the column order matches
        loaded_stats_dataframe = loaded_stats_dataframe[original_stats_dataframe.columns]

        assert original_stats_dataframe.equals(loaded_stats_dataframe)

    @pytest.mark.unit
    def test_load_axon_morphometrics_throws_error_if_file_doesnt_exist(self):

        nonExistingFolder = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        nonExistingFolder = Path(nonExistingFolder)

        with pytest.raises(FileNotFoundError):
            load_axon_morphometrics(nonExistingFolder / "dummy_name.pkl")

    @pytest.mark.unit
    def test_rearrange_column_names_for_saving_returns_columns_in_expected_order(self):
        dummy_dict = {}
        initial_columns = []
        expected_columns = []
        obtained_columns = []

        for column_name in params.column_names_ordered:
            if column_name.display_name is not None:
                name_to_use = column_name.display_name
            else:
                name_to_use = column_name.key_name
            dummy_dict[column_name.key_name] = 0.0
            expected_columns.append(name_to_use)
            initial_columns.append(column_name.key_name)

        dummy_df = pd.DataFrame(dummy_dict, index=[0])
        random.shuffle(initial_columns)
        dummy_df = dummy_df[initial_columns]
        obtained_columns = rearrange_column_names_for_saving(dummy_df).columns.to_list()

        assert expected_columns == obtained_columns

    @pytest.mark.unit
    def test_rename_column_names_after_loading_returns_renamed_columns(self):
        dummy_dict = {}
        expected_columns = []
        obtained_columns = []

        for column_name in params.column_names_ordered:
            if column_name.display_name is not None:
                name_to_use = column_name.display_name
            else:
                name_to_use = column_name.key_name
            dummy_dict[name_to_use] = 0.0
            expected_columns.append(column_name.key_name)

        dummy_df = pd.DataFrame(dummy_dict, index=[0])
        obtained_columns = rename_column_names_after_loading(dummy_df).columns.to_list()
        assert expected_columns.sort() == obtained_columns.sort()

    # --------------check consistency with reference morphometrics-------------- #
    @pytest.mark.unit
    def test_morphometrics_consistency(self):
        path_morphometrics_reference = Path(
            self.testPath /
            '__test_files__' /
            '__test_demo_files__' /
            '__morphometrics__' /
            'reference_morphometrics.pkl'
        )

        reference_stats_dataframe = load_axon_morphometrics(str(path_morphometrics_reference))
        new_stats_dataframe = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path), im_myelin=self.pred_myelin)

        for column in reference_stats_dataframe:
            column_ref_vals = reference_stats_dataframe[column].to_numpy()
            column_new_vals = new_stats_dataframe[column].to_numpy()
            assert np.allclose(column_ref_vals, column_new_vals, rtol=0, atol=1e-11, equal_nan=True)


    # --------------draw_axon_diameter tests-------------- #
    @pytest.mark.unit
    def test_draw_axon_diameter_creates_file_in_expected_location(self):
        img = ads.imread(self.test_folder_path / 'image.png')
        path_prediction = self.test_folder_path / ('image' + str(axonmyelin_suffix))

        result_path = self.test_folder_path / 'image_map-axondiameter.png'
        fig = draw_axon_diameter(img, str(path_prediction), self.pred_axon, self.pred_myelin)
        assert fig.axes
        fig.savefig(result_path)

        assert result_path.is_file()
        result_path.unlink()

    @pytest.mark.unit
    def test_draw_axon_diameter_creates_file_in_expected_location_with_axon_as_ellipse(self):
        img = ads.imread(self.test_folder_path / 'image.png')
        path_prediction = self.test_folder_path / ('image' + str(axonmyelin_suffix))

        result_path = self.test_folder_path / 'image_map-axondiameter.png'
        fig = draw_axon_diameter(img, str(path_prediction), self.pred_axon, self.pred_myelin, axon_shape=self.axon_shape)
        assert fig.axes
        fig.savefig(result_path)

        assert result_path.is_file()
        result_path.unlink()

    # --------------get_aggregate_morphometrics tests-------------- #
    @pytest.mark.unit
    def test_get_aggregate_morphometrics_returns_expected_type(self):

        aggregate_metrics = get_aggregate_morphometrics(
            self.pred_axon,
            self.pred_myelin,
            str(self.test_folder_path)
            )

        assert isinstance(aggregate_metrics, dict)

    @pytest.mark.unit
    def test_get_aggregate_morphometrics_returns_expected_type_with_axon_as_ellipse(self):

        aggregate_metrics = get_aggregate_morphometrics(
            self.pred_axon,
            self.pred_myelin,
            str(self.test_folder_path),
            axon_shape=self.axon_shape
            )

        assert isinstance(aggregate_metrics, dict)

    @pytest.mark.unit
    def test_get_aggregate_morphometrics_returns_returns_expected_keys(self):
        expectedKeys = {'avf',
                        'mvf',
                        'gratio_aggr',
                        'mean_axon_diam',
                        'mean_myelin_diam',
                        'mean_myelin_thickness',
                        'axon_density_mm2'
                        }

        aggregate_metrics = get_aggregate_morphometrics(
            self.pred_axon,
            self.pred_myelin,
            str(self.test_folder_path)
            )

        for key in list(aggregate_metrics.keys()):
            assert key in expectedKeys

    @pytest.mark.unit
    def test_get_aggregate_morphometrics_returns_returns_expected_keys_with_axon_as_ellipse(self):
        expectedKeys = {'avf',
                        'mvf',
                        'gratio_aggr',
                        'mean_axon_diam',
                        'mean_myelin_diam',
                        'mean_myelin_thickness',
                        'axon_density_mm2'
                        }

        aggregate_metrics = get_aggregate_morphometrics(
            self.pred_axon,
            self.pred_myelin,
            str(self.test_folder_path),
            axon_shape=self.axon_shape
            )

        for key in list(aggregate_metrics.keys()):
            assert key in expectedKeys

    # --------------write_aggregate_morphometrics tests-------------- #
    @pytest.mark.unit
    def test_write_aggregate_morphometrics_creates_file_in_expected_location(self):
        aggregate_metrics = get_aggregate_morphometrics(
            self.pred_axon,
            self.pred_myelin,
            self.test_folder_path
            )

        expectedFilePath = self.tmpDir / 'aggregate_morphometrics.txt'

        write_aggregate_morphometrics(str(self.tmpDir), aggregate_metrics)

        assert expectedFilePath.is_file()

    @pytest.mark.unit
    def test_write_aggregate_morphometrics_throws_error_if_folder_doesnt_exist(self):
        aggregate_metrics = get_aggregate_morphometrics(
            self.pred_axon,
            self.pred_myelin,
            self.test_folder_path
            )

        nonExistingFolder = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        nonExistingFolder = Path(nonExistingFolder)

        with pytest.raises(IOError):
            write_aggregate_morphometrics(str(nonExistingFolder), aggregate_metrics)

    @pytest.mark.unit
    def test_morphometrics_border_info_image_border_touching_flag_simulated_axons(self):
        img = ads.imread(self.image_sim_border_info_path)
        axon, myelin = ads.extract_axon_and_myelin_masks_from_image_data(img)
        stats_df = get_axon_morphometrics(
            axon,
            im_myelin=myelin,
            pixel_size=0.1,
            return_border_info=True
        )
        
        image_border_touching_col = stats_df["image_border_touching"].to_numpy()
        assert np.array_equal(image_border_touching_col, [True, False])

    @pytest.mark.unit
    def test_morphometrics_border_info_bounding_box_columns(self):
        img = ads.imread(self.image_sim_border_info_path)
        axon, myelin = ads.extract_axon_and_myelin_masks_from_image_data(img)
        stats_df = get_axon_morphometrics(
            axon,
            im_myelin=myelin,
            pixel_size=0.1,
            return_border_info=True
        )

        bbox0 = [0, 0, 267, 267]
        bbox1 = [434, 434, 567, 567]
        bbox_cols = ['bbox_min_y', 'bbox_min_x', 'bbox_max_y', 'bbox_max_x']
        bbox_computed = stats_df[bbox_cols].to_numpy()
        
        assert np.array_equal(bbox0, bbox_computed[0]) and np.array_equal(bbox1, bbox_computed[1])
