# coding: utf-8

from pathlib import Path
import imageio
import numpy as np
import os
from skimage.transform import resize
import pytest

from AxonDeepSeg.visualization.simulate_axons import *


class TestCore(object):
    def setup(self):
        self.file_path = Path("simulation_test.png")

    def teardown(self):
        if self.file_path.is_file():
            self.file_path.unlink()

    # --------------Class tests-------------- #
    @pytest.mark.single
    def test_initiate_class(self):
        obj = SimulateAxons()

        assert isinstance(obj, SimulateAxons)

    @pytest.mark.single
    def test_default_class_properties(self):
        obj = SimulateAxons()

        assert hasattr(obj, "width")
        assert hasattr(obj, "height")
        assert hasattr(obj, "origin")
        assert hasattr(obj, "image")

    # --------------Axon tests-------------- #
    @pytest.mark.single
    def test_simulate_circular_axon(self):
        axon_radius = 40
        gratio = 0.6
        myelin_thickness = axon_radius * (1 - gratio)

        obj = SimulateAxons()

        obj.generate_axon(
            axon_radius=axon_radius, center=[100, 100], gratio=gratio, plane_angle=0
        )

        # Verify value inside the axon (center)
        assert obj.image[100, 100] == 255

        # Verify value inside the axon (outer edge)
        assert obj.image[100, 100 + (axon_radius - 1)] == 255
        assert obj.image[100, 100 - (axon_radius - 1)] == 255
        assert obj.image[100 + (axon_radius - 1), 100] == 255
        assert obj.image[100 - (axon_radius - 1), 100] == 255

        # Verify value inside the myelin (inner edge)
        assert obj.image[100, 100 + (axon_radius + 1)] == 127
        assert obj.image[100, 100 - (axon_radius + 1)] == 127
        assert obj.image[100 + (axon_radius + 1), 100] == 127
        assert obj.image[100 - (axon_radius + 1), 100] == 127

        # Verify value inside the myelin (outer edge)
        assert obj.image[100, round(100 + (axon_radius + myelin_thickness - 1))] == 127
        assert obj.image[100, round(100 - (axon_radius + myelin_thickness - 1))] == 127
        assert obj.image[round(100 - (axon_radius + myelin_thickness - 1)), 100] == 127
        assert obj.image[round(100 - (axon_radius + myelin_thickness - 1)), 100] == 127

        # Verify value outside the myelin
        assert obj.image[100, round(100 + (axon_radius + myelin_thickness + 1))] == 0
        assert obj.image[100, round(100 - (axon_radius + myelin_thickness + 1))] == 0
        assert obj.image[round(100 - (axon_radius + myelin_thickness + 1)), 100] == 0
        assert obj.image[round(100 - (axon_radius + myelin_thickness + 1)), 100] == 0

    @pytest.mark.single
    def test_simulate_ellipsoidal_axon(self):
        axon_radius = 40
        gratio = 0.6
        plane_angle = 45

        myelin_thickness = axon_radius * (1 - gratio)

        obj = SimulateAxons()

        obj.generate_axon(
            axon_radius=axon_radius,
            center=[100, 100],
            gratio=gratio,
            plane_angle=plane_angle,
        )

        axon_major_axis_radius = axon_radius / np.cos(np.deg2rad(plane_angle))
        axon_minor_axis_radius = axon_radius

        myelin_major_axis_radius = axon_major_axis_radius + myelin_thickness
        myelin_minor_axis_radius = axon_minor_axis_radius + myelin_thickness

        # Verify value inside the axon (center)
        assert obj.image[100, 100] == 255

        # Verify value inside the axon (outer edge)
        assert obj.image[100, int(round(100 + (axon_major_axis_radius - 1)))] == 255
        assert obj.image[100, int(round(100 - (axon_major_axis_radius - 1)))] == 255
        assert obj.image[int(round(100 + (axon_minor_axis_radius - 1))), 100] == 255
        assert obj.image[int(round(100 - (axon_minor_axis_radius - 1))), 100] == 255

        # Verify value inside the myelin (inner edge)
        assert obj.image[100, int(round(100 + (axon_major_axis_radius + 1)))] == 127
        assert obj.image[100, int(round(100 - (axon_major_axis_radius + 1)))] == 127
        assert obj.image[int(round(100 + (axon_minor_axis_radius + 1))), 100] == 127
        assert obj.image[int(round(100 - (axon_minor_axis_radius + 1))), 100] == 127

        # Verify value inside the myelin (outer edge)
        assert obj.image[100, int(round(100 + (myelin_major_axis_radius - 1)))] == 127
        assert obj.image[100, int(round(100 - (myelin_major_axis_radius - 1)))] == 127
        assert obj.image[int(round(100 - (myelin_minor_axis_radius - 1))), 100] == 127
        assert obj.image[int(round(100 - (myelin_minor_axis_radius - 1))), 100] == 127

        # Verify value outside the myelin
        assert obj.image[100, int(round(100 + (myelin_major_axis_radius + 1)))] == 0
        assert obj.image[100, int(round(100 - (myelin_major_axis_radius + 1)))] == 0
        assert obj.image[int(round(100 - (myelin_minor_axis_radius + 1))), 100] == 0
        assert obj.image[int(round(100 - (myelin_minor_axis_radius + 1))), 100] == 0

    # --------------Save tests-------------- #
    @pytest.mark.single
    def test_saved_file_exists(self):
        axon_radius = 40
        gratio = 0.6
        myelin_thickness = axon_radius * (1 - gratio)

        obj = SimulateAxons()

        obj.generate_axon(
            axon_radius=axon_radius, center=[100, 100], gratio=gratio, plane_angle=0
        )

        obj.save(self.file_path)

        assert self.file_path.is_file()
