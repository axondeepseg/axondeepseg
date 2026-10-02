"""Lightweight batch preview dialog for segmentation results.

Shows axon/myelin/unmyelinated-axon masks alpha-blended over the source
image. A thumbnail column on the left lets you jump straight to any image in
the batch instead of stepping through a dropdown, and there's no window-per-
image spam. A button hands off to napari (a single shared viewer, reused
rather than stacked) for anyone who wants to actually edit the masks rather
than just eyeball them.
"""
from pathlib import Path

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QScrollArea
)

from AxonDeepSeg import ads_utils as ads
from AxonDeepSeg.ads_gui.backend import compose_overlay
from AxonDeepSeg.ads_gui.napari_bridge import get_viewer

_MAX_PREVIEW_SIZE = 900
_THUMB_SIZE = QSize(110, 82)


def _array_to_pixmap(rgb_array):
    h, w, _ = rgb_array.shape
    qimg = QImage(rgb_array.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # copy: qimg wraps a buffer that goes out of scope


class PreviewDialog(QDialog):
    def __init__(self, results, parent=None):
        """results: list of {"image": Path, "axon": Path|None, "myelin": Path|None, "uaxon": Path|None}."""
        super().__init__(parent)
        self.setWindowTitle("Segmentation preview")
        self._results = results
        self._overlays = [None] * len(results)  # cached full-res composited pixmaps

        self.thumb_list = QListWidget()
        self.thumb_list.setIconSize(_THUMB_SIZE)
        self.thumb_list.setFixedWidth(150)
        self.thumb_list.currentRowChanged.connect(self._show_current)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold;")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 300)
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_scroll.setWidget(self.image_label)

        image_column = QVBoxLayout()
        image_column.addWidget(self.title_label)
        image_column.addWidget(image_scroll, stretch=1)

        self.legend_label = QLabel("Axon: blue   ·   Myelin: red   ·   Unmyelinated axon: green")

        self.napari_btn = QPushButton("Open in napari")
        self.napari_btn.clicked.connect(self._open_in_napari)

        main_row = QHBoxLayout()
        main_row.addWidget(self.thumb_list)
        main_row.addLayout(image_column, stretch=1)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.legend_label)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.napari_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(main_row, stretch=1)
        layout.addLayout(bottom_row)
        self.resize(800, 620)

        self._populate_thumbnails()
        if self.thumb_list.count() > 0:
            self.thumb_list.setCurrentRow(0)

    def _populate_thumbnails(self):
        for i, entry in enumerate(self._results):
            try:
                overlay = self._compute_overlay(entry)
                self._overlays[i] = _array_to_pixmap(overlay)
                icon = QIcon(self._overlays[i].scaled(
                    _THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            except Exception:
                icon = QIcon()
            item = QListWidgetItem(icon, "")
            item.setToolTip(entry["image"].name)
            self.thumb_list.addItem(item)

    def _compute_overlay(self, entry):
        image = ads.imread(str(entry["image"]))
        axon = ads.imread(str(entry["axon"])) if entry["axon"] else None
        myelin = ads.imread(str(entry["myelin"])) if entry["myelin"] else None
        uaxon = ads.imread(str(entry["uaxon"])) if entry.get("uaxon") else None
        return compose_overlay(image, axon=axon, myelin=myelin, uaxon=uaxon)

    def _current_entry(self):
        return self._results[self.thumb_list.currentRow()]

    def _show_current(self, index):
        if index < 0 or index >= len(self._results):
            return
        self.title_label.setText(self._results[index]["image"].name)
        pixmap = self._overlays[index]
        if pixmap is None:
            self.image_label.setText(f"Could not preview {self._results[index]['image'].name}.")
            return
        shown = pixmap
        if max(pixmap.width(), pixmap.height()) > _MAX_PREVIEW_SIZE:
            shown = pixmap.scaled(
                _MAX_PREVIEW_SIZE, _MAX_PREVIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        self.image_label.setPixmap(shown)

    def _open_in_napari(self):
        entry = self._current_entry()
        try:
            viewer = get_viewer()
            viewer.layers.clear()

            image = ads.imread(str(entry["image"]))
            viewer.add_image(image, name=entry["image"].stem)
            if entry["axon"] is not None:
                axon = ads.imread(str(entry["axon"])).astype(bool)
                viewer.add_labels(axon, colormap={None: "transparent", 1: "blue"}, name="axon")
            if entry["myelin"] is not None:
                myelin = ads.imread(str(entry["myelin"])).astype(bool)
                viewer.add_labels(myelin, colormap={None: "transparent", 1: "red"}, name="myelin")
            if entry.get("uaxon") is not None:
                uaxon = ads.imread(str(entry["uaxon"])).astype(bool)
                viewer.add_labels(uaxon, colormap={None: "transparent", 1: "green"}, name="unmyelinated axon")
        except ImportError:
            self.legend_label.setText("napari is not installed.")
        except Exception as e:
            # Never let a napari hand-off failure take the whole app down with it.
            self.legend_label.setText(f"Could not open napari: {e}")
