# coding: utf-8

import pytest
import os
import inspect
import random
import string
import tempfile
import numpy as np

from AxonDeepSeg.morphometrics.compute_morphometrics import *


class TestCore(object):
    def setup(self):
        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        # Move up to the package's root directory, "axondeepseg/"
        self.packagePath = os.path.split(os.path.split(self.fullPath)[0])[0]

        self.pixelsizeFileName = os.path.join(
            self.packagePath,
            'AxonDeepSeg/data_test/pixel_size_in_micrometer.txt')
        self.pixelsizeValue = 0.07 # For current demo data.

    def teardown(self):
        pass

    #--------------get_pixelsize.py tests--------------#
    def test_get_pixelsize_returns_expected_value(self):
        expectedValue = self.pixelsizeValue
        actualValue = get_pixelsize(self.pixelsizeFileName)

        assert actualValue == expectedValue

    def test_get_pixelsize_throws_error_for_nonexisisting_file(self):
        nonExistingFile = ''.join(
            random.choice(string.lowercase) for i in range(16))

        with pytest.raises(IOError):
            get_pixelsize(nonExistingFile)

    def test_get_pixelsize_throws_error_for_invalid_data_file(self):
        with tempfile.NamedTemporaryFile() as tmp:
            # Data written using tempfile module are saved in a binary format by
            # default, which get_pixelsize doesn't currently support.
            tmp.write(repr(self.pixelsizeValue))

            with pytest.raises(ValueError):
                get_pixelsize(tmp.name)

    #--------------get_axon_morphometrics tests--------------#
    def test_get_axon_morphometrics_returns_expected_type(self):
        path_folder = self.pixelsizeFileName.split('pixel_size_in_micrometer.txt')[0]
        pred_axon = imread(os.path.join(path_folder,'AxonDeepSeg_seg-axon.png'),flatten=True)

        stats_array = get_axon_morphometrics(pred_axon,path_folder)

        assert isinstance(stats_array, np.ndarray)

    def test_get_axon_morphometrics_returns_expected_keys(self):
        expectedKeys = {'y0',
                        'x0',
                        'axon_diam',
                        'solidity',
                        'eccentricity',
                        'orientation'
                        }

        path_folder = self.pixelsizeFileName.split('pixel_size_in_micrometer.txt')[0]
        pred_axon = imread(os.path.join(path_folder,'AxonDeepSeg_seg-axon.png'),flatten=True)

        stats_array = get_axon_morphometrics(pred_axon,path_folder)

        for key in expectedKeys:
            assert key in stats_array[0]
