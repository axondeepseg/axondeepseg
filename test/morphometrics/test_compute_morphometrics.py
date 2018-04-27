# coding: utf-8

import pytest
import os
import inspect
import random
import string
import tempfile

from AxonDeepSeg.morphometrics.compute_morphometrics import get_pixelsize


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
