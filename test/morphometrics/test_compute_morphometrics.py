# coding: utf-8

from pathlib import Path
import string
import random
import math
import shutil
import numpy as np
from imageio import imread as imageio_imread  # to avoid confusion with mpl.pyplot.imread
import pytest

import AxonDeepSeg
from AxonDeepSeg.visualization.simulate_axons import SimulateAxons
from AxonDeepSeg.morphometrics.compute_morphometrics import (  
                                                                get_pixelsize,
                                                                get_axon_morphometrics, 
                                                                save_axon_morphometrics, 
                                                                load_axon_morphometrics, 
                                                                draw_axon_diameter,
                                                                save_map_of_axon_diameters,
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

        pred_axon_path = self.test_folder_path / ('image' +  str(axon_suffix))
        self.pred_axon = imageio_imread(pred_axon_path, as_gray=True)
        pred_myelin_path = self.test_folder_path / ('image' + str(myelin_suffix))
        self.pred_myelin = imageio_imread(pred_myelin_path, as_gray=True)

        self.tmpDir = self.fullPath / '__tmp__'
        if not self.tmpDir.exists():
            self.tmpDir.mkdir()

    def teardown(self):
        if self.tmpDir.exists():
            shutil.rmtree(self.tmpDir)

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
        stats_array = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))
        assert isinstance(stats_array, np.ndarray)

    @pytest.mark.unit
    def test_get_axon_morphometrics_returns_expected_keys(self):
        expectedKeys = {'y0',
                        'x0',
                        'axon_diam',
                        'axon_area',
                        'axon_perimeter',
                        'axonmyelin_perimeter',
                        'solidity',
                        'eccentricity',
                        'orientation'
                        }

        stats_array = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))

        for key in list(stats_array[0].keys()):
            assert key in expectedKeys

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask(self):
        stats_array = get_axon_morphometrics(
            self.pred_axon,
            str(self.test_folder_path),
            im_myelin=self.pred_myelin
            )
        assert stats_array[1]['gratio'] == pytest.approx(0.74, rel=0.01)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_myelin_mask_simulated_axons(self):
        path_pred = (
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons.png'
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

        # Read paths and compute axon/myelin masks
        pred = imageio_imread(path_pred)
        pred_axon = pred > 200
        pred_myelin = np.logical_and(pred >= 50, pred <= 200)

        # Compute axon morphometrics
        stats_array = get_axon_morphometrics(pred_axon, str(path_pred.parent), im_myelin=pred_myelin)

        for ii in range(0,9):
            assert stats_array[ii]['gratio'] == pytest.approx(gratio_sim[ii], rel=0.1)
            assert stats_array[ii]['axon_diam'] == pytest.approx(axon_diam_sim[ii], rel=0.1)
            assert stats_array[ii]['myelin_thickness'] == pytest.approx(myelin_thickness_sim[ii], rel=0.1)

    @pytest.mark.unit
    def test_get_axon_morphometrics_with_unexpected_myelin_mask_simulated_axons(self):
        path_pred = (
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'SimulatedAxons.png'
            )

        prediction = imageio_imread(path_pred, as_gray=True)
        pred_axon = prediction > 200
        unexpected_pred_myelin = np.zeros(prediction.shape)

        stats_array = get_axon_morphometrics(
            pred_axon,
            str(path_pred.parent),
            im_myelin=unexpected_pred_myelin
            )

        for axon_prop in stats_array:
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

        # axon perimeter (inner perimeter of myelin) = pi * diameter of axon                        
        axon_perimeter_sim = math.pi * axon_diam_sim

        prediction = imageio_imread(image_sim_path, as_gray=True)
        pred_axon = prediction > 200
        pred_myelin = (prediction > 0) & (prediction < 200)

        stats_array = get_axon_morphometrics(
            pred_axon,
            str(image_sim_path.parent),
            im_myelin=pred_myelin
            )
        
        for ii in range(0,9):
            assert stats_array[ii]['axon_perimeter'] == pytest.approx(axon_perimeter_sim[ii], rel=0.1)

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
        myelin_thickness =  (axon_diam_sim/2)  * ((1 / gratio_sim) - 1)              

        # axonmyelin perimeter (outer perimeter of myelin) = pi * diameter of axon + myelin
        axonmyelin_perimeter_sim = math.pi * ((myelin_thickness * 2) + axon_diam_sim)

        prediction = imageio_imread(image_sim_path, as_gray=True)
        pred_axon = prediction > 200
        pred_myelin = (prediction > 0) & (prediction < 200)

        stats_array = get_axon_morphometrics(
            pred_axon,
            str(image_sim_path.parent),
            im_myelin=pred_myelin
            )
        
        for ii in range(0,9):
            assert stats_array[ii]['axonmyelin_perimeter'] == pytest.approx(axonmyelin_perimeter_sim[ii], rel=0.1)

        if image_sim_path.exists():
            image_sim_path.unlink()

    # --------------save and load _axon_morphometrics tests-------------- #
    @pytest.mark.unit
    def test_save_axon_morphometrics_creates_file_in_expected_location(self):
        stats_array = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))

        save_axon_morphometrics(str(self.tmpDir), stats_array)

        # Filename 'axonlist.npy' is hardcoded in `save_axon_morphometrics()`.
        expectedFilePath = self.tmpDir / 'axonlist.npy'

        assert expectedFilePath.is_file()

    @pytest.mark.unit
    def test_save_axon_morphometrics_throws_error_if_folder_doesnt_exist(self):
        stats_array = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))

        nonExistingFolder = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        nonExistingFolder = Path(nonExistingFolder)

        with pytest.raises(IOError):
            save_axon_morphometrics(str(nonExistingFolder), stats_array)

    @pytest.mark.unit
    def test_load_axon_morphometrics_returns_identical_var_as_was_saved(self):
        original_stats_array = get_axon_morphometrics(self.pred_axon, str(self.test_folder_path))

        save_axon_morphometrics(str(self.tmpDir), original_stats_array)

        # Load method only takes in a directory as an argument, expects that
        # 'axonlist.npy' will be in directory.
        loaded_stats_array = load_axon_morphometrics(str(self.tmpDir))

        assert np.array_equal(loaded_stats_array, original_stats_array)

    @pytest.mark.unit
    def test_load_axon_morphometrics_throws_error_if_folder_doesnt_exist(self):

        nonExistingFolder = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        nonExistingFolder = Path(nonExistingFolder)

        with pytest.raises(IOError):
            load_axon_morphometrics(str(nonExistingFolder))

    # --------------draw_axon_diameter tests-------------- #
    @pytest.mark.unit
    def test_draw_axon_diameter_creates_file_in_expected_location(self):
        img = imageio_imread(self.test_folder_path / 'image.png')
        path_prediction = self.test_folder_path / ('image' + str(axonmyelin_suffix))

        result_path = self.test_folder_path / 'image_map-axondiameter.png'
        fig = draw_axon_diameter(img, str(path_prediction), self.pred_axon, self.pred_myelin)
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
