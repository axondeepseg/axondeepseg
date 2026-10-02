# coding: utf-8

import subprocess
import sys
from unittest.mock import patch

import pytest
from loguru import logger

import AxonDeepSeg.cli_deprecated as cli_deprecated

# (old command, module whose main() it should call, new command)
DEPRECATED_COMMANDS = [
    ("axondeepseg", "AxonDeepSeg.segment", "ads_segment"),
    ("axondeepseg_morphometrics", "AxonDeepSeg.morphometrics.launch_morphometrics_computation", "ads_morphometrics"),
    ("axondeepseg_aggregate", "AxonDeepSeg.morphometrics.aggregate", "ads_aggregate"),
    ("axondeepseg_filter", "AxonDeepSeg.morphometrics.filter_morphometrics", "ads_filter"),
    ("axondeepseg_count", "AxonDeepSeg.morphometrics.count_axons", "ads_count"),
    ("axondeepseg_test", "AxonDeepSeg.integrity_test", "ads_test"),
    ("download_model", "AxonDeepSeg.download_model", "ads_download_model"),
    ("download_tests", "AxonDeepSeg.download_tests", "ads_download_tests"),
]


class TestCore(object):
    def setup_method(self):
        self.messages = []
        self.sink_id = logger.add(self.messages.append, level="WARNING", format="{message}")

    def teardown_method(self):
        logger.remove(self.sink_id)

    # --------------cli_deprecated.py tests-------------- #
    @pytest.mark.unit
    @pytest.mark.parametrize("old_cmd, module, new_cmd", DEPRECATED_COMMANDS)
    def test_deprecated_command_warns_and_calls_the_new_one(self, old_cmd, module, new_cmd):
        with patch(f"{module}.main") as mock_main:
            getattr(cli_deprecated, old_cmd)()

        mock_main.assert_called_once()
        assert len(self.messages) == 1
        assert f"'{old_cmd}'" in self.messages[0]
        assert f"'{new_cmd}'" in self.messages[0]
        assert "v6" in self.messages[0]

    @pytest.mark.unit
    def test_importing_the_module_does_not_load_torch(self):
        # Run in a fresh interpreter, other tests may already have imported torch
        code = "import sys, AxonDeepSeg.cli_deprecated; print('torch' in sys.modules)"
        out = subprocess.check_output([sys.executable, "-c", code], text=True)
        assert out.strip() == "False"
