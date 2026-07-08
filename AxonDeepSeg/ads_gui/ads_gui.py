#!/usr/bin/env python3
"""Standalone GUI for AxonDeepSeg."""
import sys
from pathlib import Path

import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PyQt5 import uic

UI_FILE = Path(__file__).parent / "ads_gui.ui"
MODEL_CARDS_FILE = Path(__file__).parent.parent / "model_cards.yaml"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Mock version/release info per model key (key = "<card_key>__<variant>")
MODEL_VERSIONS = {
    "generalist__light":          {"version": "r20240416", "date": "Apr 16, 2024", "up_to_date": True},
    "generalist__ensemble":       {"version": "r20240224", "date": "Feb 24, 2024", "up_to_date": False, "latest": "r20240416"},
    "dedicated-BF__light":        {"version": "r20240416", "date": "Apr 16, 2024", "up_to_date": True},
    "dedicated-SEM__light":       {"version": "r20240403", "date": "Apr 3, 2024",  "up_to_date": True},
    "dedicated-SEM__ensemble":    {"version": "r20240403", "date": "Apr 3, 2024",  "up_to_date": True},
    "dedicated-CARS__light":      {"version": "r20240403", "date": "Apr 3, 2024",  "up_to_date": True},
    "dedicated-CARS__ensemble":   {"version": "r20240403", "date": "Apr 3, 2024",  "up_to_date": True},
    "unmyelinated-TEM__light":    {"version": "v2.0.0",    "date": "Jul 8, 2024",  "up_to_date": True},
    "unmyelinated-TEM__ensemble": {"version": "r20240708", "date": "Jul 8, 2024",  "up_to_date": True},
}

# ──────────────────────────────────────────────────────────────────
# Model catalogue
# ──────────────────────────────────────────────────────────────────

def _load_catalogue():
    with open(MODEL_CARDS_FILE) as f:
        raw = yaml.safe_load(f)

    catalogue = {}
    for key, card in raw.items():
        installed_light = (MODELS_DIR / f"{card['full_name']}_light").exists()
        installed_ensemble = (MODELS_DIR / card["full_name"]).exists()
        has_ensemble = card["weights"].get("ensemble") is not None

        entries = [{
            "key": f"{key}__light",
            "full_name": f"{card['full_name']}_light",
            "variant": "light",
            "installed": installed_light,
            "info": card.get("model-info", ""),
            "pixel_size": card.get("pixel_size"),
        }]
        if has_ensemble:
            entries.append({
                "key": f"{key}__ensemble",
                "full_name": card["full_name"],
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


# ──────────────────────────────────────────────────────────────────
# Stylesheets
# ──────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────

class ADSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)

        self._dark = True
        self._catalogue = _load_catalogue()

        self._setup_segment_tab()
        self._setup_morphometrics_tab()
        self._setup_models_tab()

        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.setEnabled(False)
        self.tabWidget.currentChanged.connect(self._on_tab_changed)

        self.theme_btn.clicked.connect(self._toggle_theme)
        self.napari_btn.clicked.connect(self._open_napari)

        self._apply_theme()

    # ── Segment tab ───────────────────────────────────────────────

    def _setup_segment_tab(self):
        self.batch_list.clear()

        self.model_combo.clear()
        self._seg_model_entries = []
        for group in self._catalogue.values():
            for entry in group["entries"]:
                tag = "  ✓" if entry["installed"] else ""
                self.model_combo.addItem(f"{group['display_name']} ({entry['variant']}){tag}")
                self._seg_model_entries.append(entry)

        self.inputBrowseBtn.clicked.connect(self._seg_browse_file)
        self.addFilesBtn.clicked.connect(self._seg_add_files)
        self.addFolderBtn.clicked.connect(self._seg_add_folder)
        self.removeBtn.clicked.connect(self._seg_remove)
        self.customBrowseBtn.clicked.connect(self._seg_browse_custom)
        self.outBrowseBtn.clicked.connect(self._seg_browse_output)

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
        if folder:
            self._seg_add_to_list(folder)

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

    def _seg_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.out_edit.setText(folder)

    # ── Morphometrics tab ─────────────────────────────────────────

    def _setup_morphometrics_tab(self):
        self.morph_batch_list.clear()
        self.morphBrowseBtn.clicked.connect(self._morph_browse)
        self.morphAddFilesBtn.clicked.connect(self._morph_add_files)
        self.morphAddFolderBtn.clicked.connect(self._morph_add_folder)
        self.morphRemoveBtn.clicked.connect(self._morph_remove)

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

    # ── Models tab ────────────────────────────────────────────────

    def _setup_models_tab(self):
        self.model_list.clear()
        self._model_tab_entries = []

        for group in self._catalogue.values():
            for entry in group["entries"]:
                tag = "  ✓" if entry["installed"] else ""
                self.model_list.addItem(f"{group['display_name']} ({entry['variant']}){tag}")
                self._model_tab_entries.append((group, entry))

        self.model_list.currentRowChanged.connect(self._on_model_selected)

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

        if entry["installed"]:
            self.status_badge.setText("✓  Installed")
        else:
            self.status_badge.setText("Not installed")

        ver = MODEL_VERSIONS.get(entry["key"], {})
        if ver:
            date = ver["date"]
            version = ver["version"]
            if ver["up_to_date"]:
                self.version_label.setText(f"{version}  ·  {date}  ·  Up to date")
            else:
                latest = ver.get("latest", "")
                self.version_label.setText(f"{version}  ·  {date}  ·  Update available ({latest})")
        else:
            self.version_label.setText("")

    def _on_download(self):
        row = self.model_list.currentRow()
        if row < 0:
            return
        group, entry = self._model_tab_entries[row]
        self._log(f"Download not yet wired — would download {group['display_name']} ({entry['variant']})")

    # ── Run / Download ────────────────────────────────────────────

    def _on_tab_changed(self, index):
        self.run_btn.setText("Download" if index == 2 else "Run")

    def _on_run(self):
        if self.tabWidget.currentIndex() == 2:
            self._on_download()
        else:
            self._log("Segmentation backend not yet wired.")

    # ── Helpers ───────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()

    # ── Theme ─────────────────────────────────────────────────────

    def _apply_theme(self):
        if self._dark:
            self.setStyleSheet(DARK)
            self.theme_btn.setText("☀  Light")
        else:
            self.setStyleSheet(LIGHT)
            self.theme_btn.setText("🌙  Dark")

    def _toggle_theme(self):
        self._dark = not self._dark
        self._apply_theme()

    # ── Napari ────────────────────────────────────────────────────

    def _open_napari(self):
        try:
            import napari
            napari.Viewer()
            self._log("Opened napari viewer.")
        except ImportError:
            QMessageBox.warning(
                self, "napari not found",
                "Install napari to use this feature:\n  pip install napari[pyqt5]"
            )


def main():
    app = QApplication(sys.argv)
    win = ADSWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
