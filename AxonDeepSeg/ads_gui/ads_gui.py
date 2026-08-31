#!/usr/bin/env python3
"""Standalone GUI for AxonDeepSeg."""
import sys
from pathlib import Path

import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import uic

from AxonDeepSeg.ads_gui.backend import (
    SegmentThread, MorphometricsThread, DownloadThread, expand_to_image_files
)
from AxonDeepSeg.ads_gui.preview import PreviewDialog

UI_FILE = Path(__file__).parent / "ads_gui.ui"
MODEL_CARDS_FILE = Path(__file__).parent.parent / "model_cards.yaml"
MODELS_DIR = Path(__file__).parent.parent / "models"
# Circles-only mark (background-removed) for the window/taskbar icon — the full
# wordmark logo below has a white background box that looks bad at icon size.
APP_ICON_FILE = Path(__file__).parent / "app_icon.png"
# Full "AxonDeepSeg" wordmark, already used elsewhere in the app (ads_napari plugin
# panel) — shown in the main window header instead of a plain text title. Dark text
# on dark theme is unreadable, so there's a white-text variant for that case.
HEADER_LOGO_FILE_DARK_THEME = Path(__file__).parent.parent / "logo_ads-alpha_white.png"
HEADER_LOGO_FILE_LIGHT_THEME = Path(__file__).parent.parent / "logo_ads-alpha.png"

# Model catalogue

def _load_catalogue():
    with open(MODEL_CARDS_FILE) as f:
        raw = yaml.safe_load(f)

    catalogue = {}
    for key, card in raw.items():
        # download_model() always names the installed folder "<full_name>_<variant>"
        # (see AxonDeepSeg/download_model.py) — both variants need the suffix here too.
        installed_light = (MODELS_DIR / f"{card['full_name']}_light").exists()
        installed_ensemble = (MODELS_DIR / f"{card['full_name']}_ensemble").exists()
        has_ensemble = card["weights"].get("ensemble") is not None

        entries = [{
            "card_key": key,
            "full_name": f"{card['full_name']}_light",
            "variant": "light",
            "installed": installed_light,
            "info": card.get("model-info", ""),
            "pixel_size": card.get("pixel_size"),
        }]
        if has_ensemble:
            entries.append({
                "card_key": key,
                "full_name": f"{card['full_name']}_ensemble",
                "variant": "ensemble",
                "installed": installed_ensemble,
                "info": card.get("model-info", ""),
                "pixel_size": card.get("pixel_size"),
            })

        catalogue[key] = {"display_name": _card_display(key), "entries": entries}
    return catalogue


def _card_display(key):
    return {
        "generalist": "Generalist",
        "dedicated-BF": "Dedicated - BF",
        "dedicated-SEM": "Dedicated - SEM",
        "dedicated-CARS": "Dedicated - CARS",
        "unmyelinated-TEM": "Unmyelinated - TEM (Stanford)",
    }.get(key, key)


def _px_display(px):
    if px is None:
        return "n/a"
    if isinstance(px, list):
        return f"{min(px)} - {max(px)} µm/px"
    return f"{px} µm/px"


# Stylesheets

DARK = """
QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 8px; padding-top: 8px; color: #cdd6f4; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QTabWidget::pane { border: 1px solid #45475a; }
QTabBar::tab { background: #313244; color: #cdd6f4; padding: 6px 16px; }
QTabBar::tab:selected { background: #45475a; }
QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px 10px; }
QPushButton:hover { background: #45475a; }
QPushButton:disabled { color: #6c7086; border-color: #313244; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 3px 6px; }
QListWidget { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; }
QListWidget::item:selected { background: #45475a; }
QTextEdit { background: #181825; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; }
QProgressBar { background: #313244; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #89b4fa; border-radius: 3px; }
QCheckBox { color: #cdd6f4; }
QScrollArea { border: none; }
QFrame[frameShape="4"] { color: #45475a; }
"""

LIGHT = """
QMainWindow, QWidget { background: #f8f8f8; color: #1e1e2e; }
QGroupBox { border: 1px solid #ccd0da; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QTabWidget::pane { border: 1px solid #ccd0da; }
QTabBar::tab { background: #e6e9ef; color: #1e1e2e; padding: 6px 16px; }
QTabBar::tab:selected { background: #ccd0da; }
QPushButton { background: #e6e9ef; color: #1e1e2e; border: 1px solid #ccd0da; border-radius: 4px; padding: 4px 10px; }
QPushButton:hover { background: #ccd0da; }
QPushButton:disabled { color: #9ca0b0; border-color: #e6e9ef; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff; color: #1e1e2e; border: 1px solid #ccd0da; border-radius: 4px; padding: 3px 6px; }
QListWidget { background: #ffffff; color: #1e1e2e; border: 1px solid #ccd0da; border-radius: 4px; }
QListWidget::item:selected { background: #ccd0da; }
QTextEdit { background: #ffffff; color: #1e1e2e; border: 1px solid #ccd0da; border-radius: 4px; }
QProgressBar { background: #e6e9ef; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #1e66f5; border-radius: 3px; }
QCheckBox { color: #1e1e2e; }
QScrollArea { border: none; }
"""


# Main window

class ADSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)

        if APP_ICON_FILE.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_FILE)))

        self._dark = True
        self._catalogue = _load_catalogue()
        self._active_thread = None
        self._preview_dialog = None

        self._setup_segment_tab()
        self._setup_morphometrics_tab()
        self._setup_models_tab()

        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.visualize_btn.clicked.connect(self._on_visualize_clicked)
        self.run_another_btn.clicked.connect(self._on_run_another_clicked)
        self._last_segment_results = []
        self.tabWidget.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(self.tabWidget.currentIndex())

        self.theme_btn.clicked.connect(self._toggle_theme)
        self.napari_btn.clicked.connect(self._open_napari)

        self.progress.setTextVisible(False)

        self._apply_theme()

    # Segment tab

    def _setup_segment_tab(self):
        self.batch_list.clear()
        self.model_hint.linkActivated.connect(lambda _: self.tabWidget.setCurrentIndex(2))
        self._populate_segment_model_combo()

        self.inputBrowseBtn.clicked.connect(self._seg_browse_file)
        self.addFilesBtn.clicked.connect(self._seg_add_files)
        self.addFolderBtn.clicked.connect(self._seg_add_folder)
        self.removeBtn.clicked.connect(self._seg_remove)
        self.customBrowseBtn.clicked.connect(self._seg_browse_custom)

        # Pixel size only matters when morphometrics is actually going to run.
        self.morpho_after.toggled.connect(self._on_morpho_after_toggled)
        self._on_morpho_after_toggled(self.morpho_after.isChecked())

    def _on_morpho_after_toggled(self, checked: bool):
        self.morphoAfterPxLabel.setVisible(checked)
        self.morpho_after_px_spin.setVisible(checked)
        self.morphoAfterPxHint.setVisible(checked)

    def _populate_segment_model_combo(self):
        current_text = self.model_combo.currentText() if self.model_combo.count() else None
        self.model_combo.clear()
        self._seg_model_entries = []
        n_missing = 0
        for group in self._catalogue.values():
            for entry in group["entries"]:
                label = f"{group['display_name']} ({entry['variant']})"
                if not entry["installed"]:
                    n_missing += 1
                    continue
                self.model_combo.addItem(label)
                self._seg_model_entries.append(entry)
                if current_text == label:
                    self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

        installed = self.model_combo.count()
        if installed == 0:
            self.model_hint.setText(
                '<a href="models">No models installed — go to the Models tab to download one</a>'
            )
            self.model_hint.setVisible(True)
        elif installed == 1 and n_missing > 0:
            plural = "s" if n_missing > 1 else ""
            self.model_hint.setText(
                f'<a href="models">{n_missing} more model{plural} available — see Models tab</a>'
            )
            self.model_hint.setVisible(True)
        else:
            self.model_hint.setText("")
            self.model_hint.setVisible(False)

    def _seg_browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select image(s)", "", "Images (*.png *.tif *.tiff *.jpg *.jpeg)"
        )
        if paths:
            self.input_edit.setText(paths[0])
            for p in paths:
                self._seg_add_to_list(p)

    def _seg_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add image files", "", "Images (*.png *.tif *.tiff *.jpg *.jpeg)"
        )
        for p in paths:
            self._seg_add_to_list(p)

    def _seg_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add folder")
        if not folder:
            return
        # Show the actual images that will be segmented rather than the bare
        # folder path — expand_to_image_files already excludes prior seg outputs
        # (masks/overlays) so we don't accidentally re-segment them.
        files = expand_to_image_files([folder])
        if not files:
            self._log(f"No segmentable images found in {folder}.")
            return
        for f in files:
            self._seg_add_to_list(str(f))

    def _seg_add_to_list(self, path: str):
        existing = {self.batch_list.item(i).text() for i in range(self.batch_list.count())}
        if path not in existing:
            self.batch_list.addItem(path)

    def _seg_remove(self):
        for item in self.batch_list.selectedItems():
            self.batch_list.takeItem(self.batch_list.row(item))

    def _seg_browse_custom(self):
        folder = QFileDialog.getExistingDirectory(self, "Select custom model folder")
        if folder:
            self.custom_edit.setText(folder)

    def _seg_collect_paths(self):
        paths = [self.batch_list.item(i).text() for i in range(self.batch_list.count())]
        if not paths and self.input_edit.text().strip():
            paths = [self.input_edit.text().strip()]
        return paths

    def _seg_resolve_model_path(self):
        custom = self.custom_edit.text().strip()
        if custom:
            return Path(custom)

        row = self.model_combo.currentIndex()
        if row < 0 or row >= len(self._seg_model_entries):
            self._log("No model installed. Go to the Models tab to download one.")
            return None
        # Every entry in this dropdown is guaranteed installed — see _populate_segment_model_combo.
        return MODELS_DIR / self._seg_model_entries[row]["full_name"]

    def _run_segmentation(self):
        paths = self._seg_collect_paths()
        if not paths:
            self._log("Add an image, a folder, or a batch queue before running.")
            return
        path_model = self._seg_resolve_model_path()
        if path_model is None:
            return

        thread = SegmentThread()
        thread.paths = paths
        thread.path_model = path_model
        thread.allow_large_images = self.large_check.isChecked()
        thread.run_morph_after = self.morpho_after.isChecked()
        thread.morph_pixel_size = self.morpho_after_px_spin.value() or None
        self._start_thread(thread)

    # Morphometrics tab

    def _setup_morphometrics_tab(self):
        self.morph_batch_list.clear()
        self.morphBrowseBtn.clicked.connect(self._morph_browse)
        self.morphAddFilesBtn.clicked.connect(self._morph_add_files)
        self.morphAddFolderBtn.clicked.connect(self._morph_add_folder)
        self.morphRemoveBtn.clicked.connect(self._morph_remove)

        # Mode checkboxes are mutually exclusive — the backend only supports one at a time.
        self._mode_checks = {
            "myelinated": self.mode_myelin,
            "unmyelinated": self.mode_unmyelin,
            "nerve": self.mode_nerve,
        }
        for mode, box in self._mode_checks.items():
            box.clicked.connect(lambda checked, m=mode: self._on_mode_checked(m, checked))
        self._on_mode_checked("myelinated", True)

    def _on_mode_checked(self, mode, checked):
        if not checked:
            # keep at least one mode selected
            self._mode_checks[mode].setChecked(True)
            return
        for m, box in self._mode_checks.items():
            box.setChecked(m == mode)
        # colorize/diameter overlay only apply to the myelinated mode
        is_myelin = mode == "myelinated"
        self.opt_colorize.setEnabled(is_myelin)
        self.opt_diameter.setEnabled(is_myelin)

    def _morph_current_mode(self):
        for mode, box in self._mode_checks.items():
            if box.isChecked():
                return mode
        return "myelinated"

    def _morph_browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select mask(s)", "", "Images (*.png *.tif *.tiff)"
        )
        if paths:
            self.morph_input_edit.setText(paths[0])
            for p in paths:
                self._morph_add_to_list(p)

    def _morph_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add mask files", "", "Images (*.png *.tif *.tiff)"
        )
        for p in paths:
            self._morph_add_to_list(p)

    def _morph_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add folder")
        if folder:
            self._morph_add_to_list(folder)

    def _morph_add_to_list(self, path: str):
        existing = {self.morph_batch_list.item(i).text() for i in range(self.morph_batch_list.count())}
        if path not in existing:
            self.morph_batch_list.addItem(path)

    def _morph_remove(self):
        for item in self.morph_batch_list.selectedItems():
            self.morph_batch_list.takeItem(self.morph_batch_list.row(item))

    def _morph_collect_paths(self):
        paths = [self.morph_batch_list.item(i).text() for i in range(self.morph_batch_list.count())]
        if not paths and self.morph_input_edit.text().strip():
            paths = [self.morph_input_edit.text().strip()]
        return paths

    def _run_morphometrics(self):
        paths = self._morph_collect_paths()
        if not paths:
            self._log("Add a mask, a folder, or a batch queue before running.")
            return

        thread = MorphometricsThread()
        thread.paths = paths
        thread.mode = self._morph_current_mode()
        thread.axon_shape = "ellipse" if self.shape_combo.currentText().startswith("ellipse") else "circle"
        thread.pixel_size = self.px_spin.value() or None
        thread.out_name = self.out_name.text().strip() or "axon_morphometrics.xlsx"
        thread.colorize = self.opt_colorize.isChecked()
        thread.diameter_overlay = self.opt_diameter.isChecked()
        self._start_thread(thread)

    # Models tab

    def _setup_models_tab(self):
        self.model_list.currentRowChanged.connect(self._on_model_selected)
        self._populate_models_list()

    def _populate_models_list(self):
        self.model_list.clear()
        self._model_tab_entries = []

        for group in self._catalogue.values():
            for entry in group["entries"]:
                tag = "  ✓" if entry["installed"] else ""
                self.model_list.addItem(f"{group['display_name']} ({entry['variant']}){tag}")
                self._model_tab_entries.append((group, entry))

        if self.model_list.count() > 0:
            self.model_list.setCurrentRow(0)

    def _on_model_selected(self, row):
        if row < 0 or row >= len(self._model_tab_entries):
            return
        group, entry = self._model_tab_entries[row]
        self.model_name.setText(f"{group['display_name']} ({entry['variant']})")
        self.model_id.setText(f"ID: {entry['full_name']}")
        self.desc_box.setPlainText(entry["info"])
        self.px_label.setText(_px_display(entry["pixel_size"]))
        self.version_label.setText("")

        if entry["installed"]:
            self.status_badge.setText("✓  Installed")
        else:
            self.status_badge.setText("Not installed")

    def _refresh_catalogue(self):
        """Reload install status from disk and repopulate the Segment/Models dropdowns
        without touching the batch queues or re-registering any signal connections."""
        self._catalogue = _load_catalogue()
        current_model_row = self.model_list.currentRow()
        self._populate_segment_model_combo()
        self._populate_models_list()
        if 0 <= current_model_row < self.model_list.count():
            self.model_list.setCurrentRow(current_model_row)

    def _run_download(self):
        row = self.model_list.currentRow()
        if row < 0:
            return
        _, entry = self._model_tab_entries[row]
        if entry["installed"]:
            self._log(f"{entry['full_name']} is already installed.")
            return

        thread = DownloadThread()
        thread.model_key = entry["card_key"]
        thread.variant = entry["variant"]
        self._start_thread(thread)

    # Run / Stop dispatch

    _RUN_BTN_LABELS = {0: "Run Segmentation", 1: "Run Morphometrics", 2: "Download"}

    def _on_tab_changed(self, index):
        self.run_btn.setText(self._RUN_BTN_LABELS.get(index, "Run"))

    def _on_run(self):
        if self._active_thread is not None:
            # run_btn is disabled while a job is active, but guard against concurrent runs anyway
            return
        index = self.tabWidget.currentIndex()
        if index == 0:
            self._run_segmentation()
        elif index == 1:
            self._run_morphometrics()
        else:
            self._run_download()

    def _on_stop(self):
        if self._active_thread is not None:
            self._active_thread.cancel_requested = True
            self.stop_btn.setEnabled(False)

    def _start_thread(self, thread):
        self._exit_post_segment_state()
        self._active_thread = thread
        thread.log.connect(self._log)
        if hasattr(thread, "progress"):
            thread.progress.connect(self._on_progress)
        thread.finished_ok.connect(self._on_thread_finished)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(hasattr(thread, "cancel_requested"))
        # Lock the tabs (inputs, model pickers, batch queues, mode toggles, ...) while
        # a job runs — the running thread already has its own copy of every parameter,
        # so this is about avoiding a confusing "did that change anything?" mid-run,
        # not correctness.
        self.tabWidget.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress_label.setText("Running…")

        thread.start()

    def _on_progress(self, current, total, desc):
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
        else:
            self.progress.setRange(0, 0)
        self.progress_label.setText(f"{desc} ({current}/{total})" if total else desc)

    def _on_thread_finished(self, success):
        self.stop_btn.setEnabled(False)
        self.tabWidget.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if success else 0)
        self.progress_label.setText("Done." if success else "Finished with errors — see log.")

        has_results = (
            success and isinstance(self._active_thread, SegmentThread)
            and self._active_thread.segmented_results
        )
        if has_results:
            self._last_segment_results = self._active_thread.segmented_results
            if self.preview_check.isChecked():
                self._open_preview(self._last_segment_results)
            self._enter_post_segment_state()
        else:
            self.run_btn.setEnabled(True)

        self._active_thread = None
        # Models/installed status may have changed (download, or a model auto-installed).
        self._refresh_catalogue()

    def _enter_post_segment_state(self):
        """After a segmentation run with results, swap Run/Stop for a more visible
        way to see the results than the easy-to-miss 'Preview results' checkbox."""
        self.run_btn.setVisible(False)
        self.stop_btn.setVisible(False)
        self.visualize_btn.setVisible(True)
        self.run_another_btn.setVisible(True)

    def _exit_post_segment_state(self):
        self.visualize_btn.setVisible(False)
        self.run_another_btn.setVisible(False)
        self.run_btn.setVisible(True)
        self.stop_btn.setVisible(True)
        self.run_btn.setEnabled(True)

    def _on_visualize_clicked(self):
        self._open_preview(self._last_segment_results)

    def _on_run_another_clicked(self):
        self._exit_post_segment_state()

    def _open_preview(self, results):
        if not results:
            self._log("Nothing to preview — no output masks were found.")
            return
        self._preview_dialog = PreviewDialog(results, parent=self)
        self._preview_dialog.show()

    # Helpers

    def _log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()

    # Theme

    def _apply_theme(self):
        if self._dark:
            self.setStyleSheet(DARK)
            self.theme_btn.setText("☀  Light")
        else:
            self.setStyleSheet(LIGHT)
            self.theme_btn.setText("🌙  Dark")
        self._apply_header_logo()

    def _apply_header_logo(self):
        logo_file = HEADER_LOGO_FILE_DARK_THEME if self._dark else HEADER_LOGO_FILE_LIGHT_THEME
        if logo_file.exists():
            pix = QPixmap(str(logo_file))
            self.titleLabel.setPixmap(pix.scaledToHeight(64, Qt.SmoothTransformation))

    def _toggle_theme(self):
        self._dark = not self._dark
        self._apply_theme()

    # Napari

    def _open_napari(self):
        try:
            from AxonDeepSeg.ads_gui.napari_bridge import get_viewer
            get_viewer()
            self._log("Opened napari viewer.")
        except ImportError:
            QMessageBox.warning(
                self, "napari not found",
                "Install napari to use this feature:\n  pip install napari[pyqt5]"
            )


def main():
    if sys.platform == "win32":
        # Without this, Windows groups the app under python.exe's own icon in the
        # taskbar instead of using the window icon set below.
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AxonDeepSeg.ADS.StandaloneGUI")
        except Exception:
            pass

    app = QApplication(sys.argv)
    if APP_ICON_FILE.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_FILE)))
    win = ADSWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
