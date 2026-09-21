"""Widget tests for the standalone GUI's main window.

These drive the real ADSWindow through its signal/slot wiring, but never start a
segmentation — no model, no GPU and no test data needed, so they stay fast enough
to run on every platform in the CI matrix.

Note on visibility assertions: a child of a window that was never show()n always
reports isVisible() == False, so these check isHidden() instead, which tracks the
explicit setVisible() state regardless of whether the parent is on screen.
"""
from unittest.mock import patch

import pytest

from AxonDeepSeg.ads_gui import ads_gui as ads_gui_module
from AxonDeepSeg.ads_gui.ads_gui import (
    APP_ICON_FILE,
    HEADER_LOGO_FILE_DARK_THEME,
    HEADER_LOGO_FILE_LIGHT_THEME,
    ADSWindow,
)

SEGMENT_TAB, MORPHOMETRICS_TAB, MODELS_TAB = 0, 1, 2


@pytest.fixture
def window(qtbot):
    win = ADSWindow()
    qtbot.addWidget(win)
    return win


class TestRunButtonLabel:
    @pytest.mark.integration
    def test_label_matches_the_active_tab_on_startup(self, window):
        """currentChanged doesn't fire for the tab that's already active when the
        signal is connected, so the label has to be primed explicitly at startup."""
        expected = window._RUN_BTN_LABELS[window.tabWidget.currentIndex()]

        assert window.run_btn.text() == expected

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "tab, expected",
        [
            (SEGMENT_TAB, "Run Segmentation"),
            (MORPHOMETRICS_TAB, "Run Morphometrics"),
            (MODELS_TAB, "Download"),
        ],
    )
    def test_label_follows_tab_switches(self, window, tab, expected):
        window.tabWidget.setCurrentIndex(tab)

        assert window.run_btn.text() == expected

    @pytest.mark.integration
    def test_label_is_restored_when_switching_back(self, window):
        window.tabWidget.setCurrentIndex(MORPHOMETRICS_TAB)
        window.tabWidget.setCurrentIndex(SEGMENT_TAB)

        assert window.run_btn.text() == "Run Segmentation"


class TestSegmentBatchQueue:
    @staticmethod
    def _items(window):
        return [window.batch_list.item(i).text() for i in range(window.batch_list.count())]

    @pytest.mark.integration
    def test_adding_a_folder_queues_the_images_not_the_folder(self, window, tmp_path):
        """The queue shows what will actually be segmented, and prior segmentation
        outputs in the same folder must not be queued for re-segmentation."""
        for name in [
            "image.png",
            "image_seg-axon.png",
            "image_seg-myelin.png",
            "image_seg-axonmyelin.png",
            "image_grayscale.png",
            "notes.txt",
        ]:
            (tmp_path / name).touch()

        with patch.object(
            ads_gui_module.QFileDialog, "getExistingDirectory", return_value=str(tmp_path)
        ):
            window._seg_add_folder()

        assert self._items(window) == [str(tmp_path / "image.png")]

    @pytest.mark.integration
    def test_empty_folder_reports_instead_of_queueing_nothing(self, window, tmp_path):
        with patch.object(
            ads_gui_module.QFileDialog, "getExistingDirectory", return_value=str(tmp_path)
        ):
            window._seg_add_folder()

        assert self._items(window) == []
        assert "No segmentable images found" in window.log.toPlainText()

    @pytest.mark.integration
    def test_cancelled_folder_dialog_is_a_no_op(self, window):
        with patch.object(
            ads_gui_module.QFileDialog, "getExistingDirectory", return_value=""
        ):
            window._seg_add_folder()

        assert self._items(window) == []
        assert "No segmentable images found" not in window.log.toPlainText()

    @pytest.mark.integration
    def test_the_same_image_is_never_queued_twice(self, window, tmp_path):
        (tmp_path / "image.png").touch()

        with patch.object(
            ads_gui_module.QFileDialog, "getExistingDirectory", return_value=str(tmp_path)
        ):
            window._seg_add_folder()
            window._seg_add_folder()

        assert self._items(window) == [str(tmp_path / "image.png")]

    @pytest.mark.integration
    def test_remove_takes_the_selected_image_out_of_the_queue(self, window, tmp_path):
        (tmp_path / "a.png").touch()
        (tmp_path / "b.png").touch()

        with patch.object(
            ads_gui_module.QFileDialog, "getExistingDirectory", return_value=str(tmp_path)
        ):
            window._seg_add_folder()
        window.batch_list.setCurrentRow(0)
        window._seg_remove()

        assert self._items(window) == [str(tmp_path / "b.png")]

    @pytest.mark.integration
    def test_falls_back_to_the_single_input_field_when_the_queue_is_empty(self, window):
        window.input_edit.setText("  /some/image.png  ")

        assert window._seg_collect_paths() == ["/some/image.png"]


class TestMorphometricsModes:
    @pytest.mark.integration
    def test_modes_are_mutually_exclusive(self, window):
        """The backend runs one mode at a time, so checking one must clear the others."""
        window.mode_unmyelin.click()

        assert window._morph_current_mode() == "unmyelinated"
        assert window.mode_unmyelin.isChecked()
        assert not window.mode_myelin.isChecked()
        assert not window.mode_nerve.isChecked()

    @pytest.mark.integration
    def test_a_mode_cannot_be_left_unselected(self, window):
        window.mode_myelin.click()  # already checked, so this tries to uncheck it

        assert window.mode_myelin.isChecked()
        assert window._morph_current_mode() == "myelinated"

    @pytest.mark.integration
    def test_colorize_and_diameter_overlay_are_myelinated_only(self, window):
        assert window.opt_colorize.isEnabled()
        assert window.opt_diameter.isEnabled()

        window.mode_nerve.click()

        assert not window.opt_colorize.isEnabled()
        assert not window.opt_diameter.isEnabled()

        window.mode_myelin.click()

        assert window.opt_colorize.isEnabled()
        assert window.opt_diameter.isEnabled()


class TestSegmentTabPixelSizeRow:
    @pytest.mark.integration
    def test_pixel_size_row_matches_the_checkbox_at_startup(self, window):
        """Same priming problem as the run button — the row's initial visibility has
        to be set explicitly, not left to the toggled signal."""
        hidden = not window.morpho_after.isChecked()

        assert window.morphoAfterPxLabel.isHidden() == hidden
        assert window.morpho_after_px_spin.isHidden() == hidden
        assert window.morphoAfterPxHint.isHidden() == hidden

    @pytest.mark.integration
    def test_pixel_size_row_follows_the_checkbox(self, window):
        window.morpho_after.setChecked(True)

        assert not window.morphoAfterPxLabel.isHidden()
        assert not window.morpho_after_px_spin.isHidden()

        window.morpho_after.setChecked(False)

        assert window.morphoAfterPxLabel.isHidden()
        assert window.morpho_after_px_spin.isHidden()


class TestPostSegmentState:
    @pytest.mark.integration
    def test_results_swap_run_stop_for_visualize_and_run_another(self, window):
        window._enter_post_segment_state()

        assert window.run_btn.isHidden()
        assert window.stop_btn.isHidden()
        assert not window.visualize_btn.isHidden()
        assert not window.run_another_btn.isHidden()

    @pytest.mark.integration
    def test_run_another_restores_the_run_button(self, window):
        window._enter_post_segment_state()
        window._on_run_another_clicked()

        assert not window.run_btn.isHidden()
        assert window.run_btn.isEnabled()
        assert not window.stop_btn.isHidden()
        assert window.visualize_btn.isHidden()
        assert window.run_another_btn.isHidden()

    @pytest.mark.integration
    def test_preview_with_no_results_reports_instead_of_opening_a_dialog(self, window):
        window._open_preview([])

        assert "Nothing to preview" in window.log.toPlainText()
        assert window._preview_dialog is None


class TestTheme:
    @pytest.mark.integration
    def test_toggling_swaps_the_stylesheet_and_the_button_label(self, window):
        assert window._dark is True
        assert window.styleSheet() == ads_gui_module.DARK
        assert "Light" in window.theme_btn.text()

        window._toggle_theme()

        assert window._dark is False
        assert window.styleSheet() == ads_gui_module.LIGHT
        assert "Dark" in window.theme_btn.text()

    @pytest.mark.integration
    def test_each_theme_loads_a_real_header_logo(self, window):
        """Guards against the logo assets being renamed or dropped from the package —
        a missing file would silently leave the header blank at runtime."""
        for _ in range(2):
            pixmap = window.titleLabel.pixmap()
            assert pixmap is not None and not pixmap.isNull()
            window._toggle_theme()


class TestPackagedAssets:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "asset",
        [APP_ICON_FILE, HEADER_LOGO_FILE_DARK_THEME, HEADER_LOGO_FILE_LIGHT_THEME],
    )
    def test_asset_ships_with_the_package(self, asset):
        assert asset.exists(), f"{asset.name} is missing from the installed package"
