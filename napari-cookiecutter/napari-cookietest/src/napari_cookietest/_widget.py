"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QComboBox

if TYPE_CHECKING:
    import napari


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Example button")
        btn.clicked.connect(self._on_click)

        load_image_button = QPushButton("Load image")
        load_image_button.clicked.connect(self._on_load_image_button_click)

        model_selection_combobox = QComboBox()
        model_selection_combobox.addItems(["Select the model","SEM", "TEM", "BF"])

        apply_model_button = QPushButton("Apply ADS model")
        fill_axons_button = QPushButton("Fill axons")
        compute_morphometrics_button = QPushButton("Compute morphometrics")

        self.setLayout(QVBoxLayout())
        # self.layout().addWidget(btn)
        self.layout().addWidget(load_image_button)
        self.layout().addWidget(model_selection_combobox)
        self.layout().addWidget(apply_model_button)
        self.layout().addWidget(fill_axons_button)
        self.layout().addWidget(compute_morphometrics_button)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")

    def _on_load_image_button_click(self):
        # Change the path of the image to load it in Napari
        self.viewer.open("C:/Users/Stoyan/Desktop/ADS/ads_with_napari/axondeepseg"
                    "/AxonDeepSeg/models/model_seg_rat_axon-myelin_sem/data_test/image.png")
        pass


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
