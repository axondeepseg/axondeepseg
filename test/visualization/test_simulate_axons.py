# coding: utf-8

from pathlib import Path
import imageio
import numpy as np
import os
from skimage.transform import resize
import pytest

from ads_base.visualization.simulate_axons import SimulateAxons, calc_myelin_thickness


class TestCore(object):
    def setup_method(self):
        self.file_path = Path("simulation_test.png")

    def teardown_method(self):
        if self.file_path.is_file():
            self.file_path.unlink()

    # --------------Function tests-------------- #
    @pytest.mark.unit
    def test_calc_myelin_thickness(self):
        gratio = 0.6
        axon_radius = 40

        myelin_thickness = calc_myelin_thickness(axon_radius, gratio)

        actual_gratio = axon_radius / (axon_radius + myelin_thickness)

        assert actual_gratio == gratio

    # --------------Class tests-------------- #
    @pytest.mark.unit
    def test_initiate_class(self):
        obj = SimulateAxons()

        assert isinstance(obj, SimulateAxons)

    @pytest.mark.unit
    def test_default_class_properties(self):
        obj = SimulateAxons()

        assert hasattr(obj, "width")
        assert hasattr(obj, "height")
        assert hasattr(obj, "origin")
        assert hasattr(obj, "image")

    # --------------Axon tests-------------- #
    @pytest.mark.unit
    def test_simulate_circular_axon(self):
        x_pos = 100
        y_pos = 100

        axon_radius = 40
        gratio = 0.6
        myelin_thickness = calc_myelin_thickness(axon_radius, gratio)

        obj = SimulateAxons()

        obj.generate_axon(
            axon_radius=axon_radius, center=[x_pos, y_pos], gratio=gratio, plane_angle=0
        )

        # Verify value inside the axon (center)
        assert obj.image[x_pos, y_pos] == 255

        # Verify value inside the axon (outer edge)
        assert obj.image[x_pos, y_pos + (axon_radius - 1)] == 255
        assert obj.image[x_pos, y_pos - (axon_radius - 1)] == 255
        assert obj.image[x_pos + (axon_radius - 1), y_pos] == 255
        assert obj.image[x_pos - (axon_radius - 1), y_pos] == 255

        # Verify value inside the myelin (inner edge)
        assert obj.image[x_pos, y_pos + (axon_radius + 1)] == 127
        assert obj.image[x_pos, y_pos - (axon_radius + 1)] == 127
        assert obj.image[x_pos + (axon_radius + 1), y_pos] == 127
        assert obj.image[x_pos - (axon_radius + 1), y_pos] == 127

        # Verify value inside the myelin (outer edge)
        assert (
            obj.image[x_pos, round(y_pos + (axon_radius + myelin_thickness - 1))] == 127
        )
        assert (
            obj.image[x_pos, round(y_pos - (axon_radius + myelin_thickness - 1))] == 127
        )
        assert (
            obj.image[round(x_pos - (axon_radius + myelin_thickness - 1)), y_pos] == 127
        )
        assert (
            obj.image[round(x_pos - (axon_radius + myelin_thickness - 1)), y_pos] == 127
        )

        # Verify value outside the myelin
        assert (
            obj.image[x_pos, round(y_pos + (axon_radius + myelin_thickness + 1))] == 0
        )
        assert (
            obj.image[x_pos, round(y_pos - (axon_radius + myelin_thickness + 1))] == 0
        )
        assert (
            obj.image[round(x_pos - (axon_radius + myelin_thickness + 1)), y_pos] == 0
        )
        assert (
            obj.image[round(x_pos - (axon_radius + myelin_thickness + 1)), y_pos] == 0
        )

    @pytest.mark.unit
    def test_simulate_ellipsoidal_axon(self):
        x_pos = 500
        y_pos = 500

        axon_radius = 40
        gratio = 0.6
        plane_angle = 70

        myelin_thickness = calc_myelin_thickness(axon_radius, gratio)

        obj = SimulateAxons()

        obj.generate_axon(
            axon_radius=axon_radius,
            center=[x_pos, y_pos],
            gratio=gratio,
            plane_angle=plane_angle,
        )

        axon_major_axis_radius = axon_radius / np.cos(np.deg2rad(plane_angle))
        axon_minor_axis_radius = axon_radius

        myelin_major_axis_radius = (axon_radius + myelin_thickness) / np.cos(
            np.deg2rad(plane_angle)
        )
        myelin_minor_axis_radius = axon_radius + myelin_thickness

        # Verify value inside the axon (center)
        assert obj.image[x_pos, y_pos] == 255

        # Verify value inside the axon (outer edge)
        assert obj.image[x_pos, int(round(y_pos + (axon_major_axis_radius - 1)))] == 255
        assert obj.image[x_pos, int(round(y_pos - (axon_major_axis_radius - 1)))] == 255
        assert obj.image[int(round(x_pos + (axon_minor_axis_radius - 1))), y_pos] == 255
        assert obj.image[int(round(x_pos - (axon_minor_axis_radius - 1))), y_pos] == 255

        # Verify value inside the myelin (inner edge)
        assert obj.image[x_pos, int(round(y_pos + (axon_major_axis_radius + 1)))] == 127
        assert obj.image[x_pos, int(round(y_pos - (axon_major_axis_radius + 1)))] == 127
        assert obj.image[int(round(x_pos + (axon_minor_axis_radius + 1))), y_pos] == 127
        assert obj.image[int(round(x_pos - (axon_minor_axis_radius + 1))), y_pos] == 127

        # Verify value inside the myelin (outer edge)
        assert (
            obj.image[x_pos, int(round(y_pos + (myelin_major_axis_radius - 1)))] == 127
        )
        assert (
            obj.image[x_pos, int(round(y_pos - (myelin_major_axis_radius - 1)))] == 127
        )
        assert (
            obj.image[int(round(x_pos - (myelin_minor_axis_radius - 1))), y_pos] == 127
        )
        assert (
            obj.image[int(round(x_pos - (myelin_minor_axis_radius - 1))), y_pos] == 127
        )

        # Verify value outside the myelin
        assert obj.image[x_pos, int(round(y_pos + (myelin_major_axis_radius + 1)))] == 0
        assert obj.image[x_pos, int(round(y_pos - (myelin_major_axis_radius + 1)))] == 0
        assert obj.image[int(round(x_pos - (myelin_minor_axis_radius + 1))), y_pos] == 0
        assert obj.image[int(round(x_pos - (myelin_minor_axis_radius + 1))), y_pos] == 0

    # --------------Save tests-------------- #
    @pytest.mark.unit
    def test_saved_file_exists(self):
        axon_radius = 40
        gratio = 0.6
        myelin_thickness = calc_myelin_thickness(axon_radius, gratio)

        obj = SimulateAxons()

        obj.generate_axon(
            axon_radius=axon_radius, center=[100, 100], gratio=gratio, plane_angle=0
        )

        obj.save(self.file_path)

        assert self.file_path.is_file()
