# coding: utf-8

from pathlib import Path
import pytest

from AxonDeepSeg.apply_model import (
    get_checkpoint_name, 
)

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

    def teardown_method(self):
        pass

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
