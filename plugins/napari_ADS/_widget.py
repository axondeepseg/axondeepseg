from typing import TYPE_CHECKING

import os, sys
from pathlib import Path

import AxonDeepSeg
import config
import numpy as np
import qtpy.QtCore
from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QComboBox, QFileDialog, QLabel, QPlainTextEdit, \
    QInputDialog, QMessageBox
from qtpy.QtCore import QStringListModel, QObject, Signal
from qtpy.QtGui import QPixmap

from AxonDeepSeg import ads_utils, segment, postprocessing, params
import AxonDeepSeg.morphometrics.compute_morphometrics as compute_morphs
from config import axonmyelin_suffix, axon_suffix, myelin_suffix

import napari
from napari.utils.notifications import show_info
from .settings_menu_ui import Ui_Settings_menu_ui

class ADSsettings:
    """
    This class handles everything related to the parameters used in the ADS plugin, including the frame for the settings
    menu.
    """
    def __init__(self, ads_plugin):
        """
        Constructor for the ADSsettings class. Initializes the default settings.
        """
        self.ads_plugin = ads_plugin

        # Declare the settings used
        self.overlap_value = 48
        self.zoom_factor = 1.0
        self.axon_shape = "circle"
        self._axon_shape_selection_index = 0
        self.no_patch = False
        self.gpu_id = 0
        self.n_gpus = ads_utils.check_available_gpus(None)
        self.max_gpu_id = self.n_gpus-1 if self.n_gpus > 0 else 0
        self.setup_settings_menu()

    def setup_settings_menu(self):
        self.Settings_menu_ui = QtWidgets.QDialog(self.ads_plugin)
        self.ui = Ui_Settings_menu_ui()
        self.ui.setupUi(self.Settings_menu_ui)
        self.ui.done_button.clicked.connect(self._on_done_button_click)

        self.ui.overlap_value_spinBox.valueChanged.connect(self._on_overlap_value_changed)
        self.ui.zoom_factor_spinBox.valueChanged.connect(self._on_zoom_factor_changed)
        self.ui.axon_shape_comboBox.currentIndexChanged.connect(self._on_axon_shape_changed)
        self.ui.no_patch_checkBox.stateChanged.connect(self._on_no_patch_changed)
        self.ui.gpu_id_spinBox.valueChanged.connect(self._on_gpu_id_changed)

    def create_settings_menu(self):
        self.ui.overlap_value_spinBox.setValue(self.overlap_value)
        self.ui.zoom_factor_spinBox.setValue(self.zoom_factor)
        self.ui.axon_shape_comboBox.setCurrentIndex(self._axon_shape_selection_index)
        self.ui.no_patch_checkBox.setChecked(self.no_patch)
        self.ui.gpu_id_spinBox.setValue(self.gpu_id)
        self.ui.gpu_id_spinBox.setMaximum(self.max_gpu_id)
        self.Settings_menu_ui.show()

    def _on_done_button_click(self):
        self.Settings_menu_ui.close()

    def _on_overlap_value_changed(self):
        self.overlap_value = self.ui.overlap_value_spinBox.value()

    def _on_zoom_factor_changed(self):
        self.zoom_factor = self.ui.zoom_factor_spinBox.value()

    def _on_axon_shape_changed(self):
        self.axon_shape = self.ui.axon_shape_comboBox.currentText()
        self._axon_shape_selection_index = self.ui.axon_shape_comboBox.currentIndex()

    def _on_no_patch_changed(self):
        self.no_patch = self.ui.no_patch_checkBox.isChecked()

    def _on_gpu_id_changed(self):
        self.gpu_id = self.ui.gpu_id_spinBox.value()


class ADSplugin(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = ADSsettings(self)

        citation_textbox = QPlainTextEdit(self)
        citation_textbox.setPlainText(self.get_citation_string())
        citation_textbox.setReadOnly(True)
        citation_textbox.setMaximumHeight(100)

        hyperlink_label = QLabel()
        hyperlink_label.setOpenExternalLinks(True)
        hyperlink_label.setText(
            '<a href="https://axondeepseg.readthedocs.io/en/latest/">Need help? Read the documentation</a>')

        self.available_models = ads_utils.get_existing_models_list()
        self.model_selection_combobox = QComboBox()
        self.model_selection_combobox.addItems(["Select the model"] + self.available_models)

        self.apply_model_button = QPushButton("Apply ADS model")
        self.apply_model_button.clicked.connect(self._on_apply_model_button_click)
        self.apply_model_thread = ApplyModelThread()
        self.apply_model_thread.model_applied_signal.connect(self._on_model_finished_apply)

        load_mask_button = QPushButton("Load mask")
        load_mask_button.clicked.connect(self._on_load_mask_button_click)

        fill_axons_button = QPushButton("Fill axons")
        fill_axons_button.clicked.connect(self._on_fill_axons_click)

        save_segmentation_button = QPushButton("Save segmentation")
        save_segmentation_button.clicked.connect(self._on_save_segmentation_button)

        compute_morphometrics_button = QPushButton("Compute morphometrics")
        compute_morphometrics_button.clicked.connect(self._on_compute_morphometrics_button)

        settings_menu_button = QPushButton("Settings")
        settings_menu_button.clicked.connect(self._on_settings_menu_clicked)

        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(10, 20, 20, 10)
        self.layout().addWidget(self.get_logo())
        self.layout().addWidget(citation_textbox)
        self.layout().addWidget(hyperlink_label)
        self.layout().addWidget(self.model_selection_combobox)
        self.layout().addWidget(self.apply_model_button)
        self.layout().addWidget(load_mask_button)
        self.layout().addWidget(fill_axons_button)
        self.layout().addWidget(save_segmentation_button)
        self.layout().addWidget(compute_morphometrics_button)
        self.layout().addWidget(settings_menu_button)
        self.layout().addStretch()

    def try_to_get_pixel_size_of_layer(self, layer):
        image_path = Path(layer.source.path)
        image_directory = image_path.parents[0]

        # Check if the pixel size txt file exist in the image_directory
        pixel_size_exists = (image_directory / "pixel_size_in_micrometer.txt").exists()

        if pixel_size_exists:
            resolution_file = open(str((image_directory / "pixel_size_in_micrometer.txt")), 'r')
            pixel_size_float = float(resolution_file.read())
            return pixel_size_float
        else:
            return None

    def add_layer_pixel_size_to_metadata(self, layer):
        pixel_size = self.try_to_get_pixel_size_of_layer(layer)
        if pixel_size is not None:
            layer.metadata["pixel_size"] = pixel_size
            return True
        else:
            return False


    def _on_apply_model_button_click(self):
        selected_layers = self.viewer.layers.selection
        selected_model = self.model_selection_combobox.currentText()

        if selected_model not in self.available_models:
            self.show_info_message("No model selected")
            return
        else:
            ads_path = Path(AxonDeepSeg.__file__).parents[0]
            model_path = ads_path / "models" / selected_model
        if len(selected_layers) != 1:
            self.show_info_message("No single image selected")
            return
        selected_layer = selected_layers.active
        image_directory = Path(selected_layer.source.path).parents[0]

        # Check if the pixel size txt file exist in the imageDirPath
        if "pixel_size" not in selected_layer.metadata.keys():
            if not self.add_layer_pixel_size_to_metadata(selected_layer):
                pixel_size = self.get_pixel_size_with_prompt()
                if pixel_size is None:
                    return
                selected_layer.metadata["pixel_size"] = pixel_size

        self.apply_model_button.setEnabled(False)
        self.apply_model_thread.selected_layer = selected_layer
        self.apply_model_thread.image_directory = image_directory
        self.apply_model_thread.path_testing_image = Path(selected_layer.source.path)
        self.apply_model_thread.path_model = model_path
        self.apply_model_thread.overlap_value = [self.settings.overlap_value, self.settings.overlap_value]
        self.apply_model_thread.zoom_factor = self.settings.zoom_factor
        self.apply_model_thread.no_patch = self.settings.no_patch
        self.apply_model_thread.gpu_id = self.settings.gpu_id
        show_info("Applying ADS model... This can take a few seconds. Check the console for more information.")
        self.apply_model_thread.start()


    def _on_model_finished_apply(self):
        self.apply_model_button.setEnabled(True)
        if not self.apply_model_thread.task_finished_successfully:
            self.show_info_message("Couldn't apply the ADS model. Check the console for more information")
            return

        selected_layer = self.apply_model_thread.selected_layer
        image_directory = self.apply_model_thread.image_directory
        image_name_no_extension = selected_layer.name
        axon_mask_path = image_directory / (image_name_no_extension + str(axon_suffix))
        axon_mask_name = image_name_no_extension + axon_suffix.stem
        myelin_mask_path = image_directory / (image_name_no_extension + str(myelin_suffix))
        myelin_mask_name = image_name_no_extension + myelin_suffix.stem

        axon_data = ads_utils.imread(axon_mask_path).astype(bool)
        self.viewer.add_labels(axon_data, color={1: 'blue'}, name=axon_mask_name,
                               metadata={"associated_image_name": image_name_no_extension})
        myelin_data = ads_utils.imread(myelin_mask_path).astype(bool)
        self.viewer.add_labels(myelin_data, color={1: 'red'}, name=myelin_mask_name,
                               metadata={"associated_image_name": image_name_no_extension})
        selected_layer.metadata["associated_axon_mask_name"] = axon_mask_name
        selected_layer.metadata["associated_myelin_mask_name"] = myelin_mask_name

    def _on_load_mask_button_click(self):
        microscopy_image_layer = self.get_microscopy_image()
        if microscopy_image_layer is None:
            self.show_info_message("No single image selected/detected")
            return

        mask_file_path, _ = QFileDialog.getOpenFileName(self, "Select the mask you wish to load")
        if mask_file_path == "":
            return

        if not self.show_ok_cancel_message("The mask will be associated with " + microscopy_image_layer.name):
            return

        img_png2D = ads_utils.imread(mask_file_path)
        # Extract the Axon mask
        axon_data = img_png2D > 200
        axon_data = axon_data.astype(np.uint8)
        axon_mask_name = microscopy_image_layer.name + config.axon_suffix.stem
        # Extract the Myelin mask
        myelin_data = (img_png2D > 100) & (img_png2D < 200)
        myelin_data = myelin_data.astype(np.uint8)
        myelin_mask_name = microscopy_image_layer.name + config.myelin_suffix.stem

        # Load the masks and add metadata to the files to keep a link between them
        self.viewer.add_labels(axon_data, color={1: 'blue'}, name=axon_mask_name,
                               metadata={"associated_image_name": microscopy_image_layer.name})
        self.viewer.add_labels(myelin_data, color={1: 'red'}, name=myelin_mask_name,
                               metadata={"associated_image_name": microscopy_image_layer.name})
        microscopy_image_layer.metadata["associated_axon_mask_name"] = axon_mask_name
        microscopy_image_layer.metadata["associated_myelin_mask_name"] = myelin_mask_name

    def _on_fill_axons_click(self):
        axon_layer = self.get_axon_layer()
        myelin_layer = self.get_myelin_layer()

        if (axon_layer is None) or (myelin_layer is None):
            self.show_info_message("One or more masks missing")
            return

        myelin_array = np.array(myelin_layer.data, copy=True)
        axon_extracted_array = postprocessing.fill_myelin_holes(myelin_array)
        axon_array_indexes = np.where(axon_extracted_array > 0)
        axon_layer._save_history((axon_array_indexes,
                                  np.array(axon_layer.data[axon_array_indexes], copy=True),
                                  1))
        axon_layer.data[axon_array_indexes] = 1
        axon_layer.refresh()

    def _on_save_segmentation_button(self):
        axon_layer = self.get_axon_layer()
        myelin_layer = self.get_myelin_layer()

        if (axon_layer is None) or (myelin_layer is None):
            self.show_info_message("One or more masks missing")
            return
        save_path = QFileDialog.getExistingDirectory(self, "Select where the segmentation should be saved")
        save_path = Path(save_path)

        # Scale the pixel values of the masks to 255 for image saving
        myelin_array = myelin_layer.data * params.intensity['binary']
        axon_array = axon_layer.data * params.intensity['binary']

        myelin_and_axon_array = (myelin_array // 2 + axon_array).astype(np.uint8)

        microscopy_image_name = axon_layer.metadata["associated_image_name"]
        axon_image_name = microscopy_image_name + str(config.axon_suffix)
        myelin_image_name = microscopy_image_name + str(config.myelin_suffix)
        axonmyelin_image_name = microscopy_image_name + str(config.axonmyelin_suffix)

        ads_utils.imwrite(filename=save_path / axonmyelin_image_name, img=myelin_and_axon_array)
        ads_utils.imwrite(filename=save_path / myelin_image_name, img=myelin_array)
        ads_utils.imwrite(filename=save_path / axon_image_name, img=axon_array)

    def _on_compute_morphometrics_button(self):
        axon_layer = self.get_axon_layer()
        myelin_layer = self.get_myelin_layer()
        microscopy_image_layer = self.get_microscopy_image()

        if (axon_layer is None) or (myelin_layer is None) or (microscopy_image_layer is None):
            self.show_info_message("Image or mask(s) missing.")
            return
        axon_data = axon_layer.data
        myelin_data = myelin_layer.data


        # Try to find the pixel size
        if "pixel_size" not in microscopy_image_layer.metadata.keys():
            pixel_size = self.get_pixel_size_with_prompt()
        else:
            pixel_size = microscopy_image_layer.metadata["pixel_size"]

        if pixel_size is None:
            return

        # Ask the user where to save
        default_name = Path(os.getcwd()) / "Morphometrics.csv"
        file_name, selected_filter = QFileDialog.getSaveFileName(self, caption="Select where to save morphometrics",
                                                                 directory=str(default_name), filter= "CSV file(*.csv)")
        if file_name == "":
            return

        # Compute statistics
        stats_dataframe, index_image_array = compute_morphs.get_axon_morphometrics(im_axon=axon_data,
                                                                                   im_myelin=myelin_data,
                                                                                   pixel_size=pixel_size,
                                                                                   axon_shape=self.settings.axon_shape,
                                                                                   return_index_image=True)
        try:
            compute_morphs.save_axon_morphometrics(file_name, stats_dataframe)

        except IOError:
            self.show_info_message("Cannot save morphometrics")

        self.viewer.add_image(data = index_image_array, rgb=False, colormap="yellow", blending="additive",
                              name="numbers")

    def _on_settings_menu_clicked(self):
        self.settings.create_settings_menu()

    def get_layer_by_name(self, name_of_layer):
        for layer in self.viewer.layers:
            if layer.name == name_of_layer:
                return layer

    def get_microscopy_image(self):
        selected_layers = self.viewer.layers.selection
        if len(selected_layers) == 0:
            return None
        selected_layer = selected_layers.active
        if selected_layer is None:
            return None

        if selected_layer.__class__ == napari.layers.image.image.Image:
            return selected_layer
        elif selected_layer.__class__ == napari.layers.labels.labels.Labels:
            return self.get_layer_by_name(selected_layer.metadata["associated_image_name"])
        else:
            return None

    def get_mask_layer(self, type_of_mask):
        selected_layers = self.viewer.layers.selection
        if len(selected_layers) == 0:
            return None
        selected_layer = selected_layers.active
        if selected_layer is None:
            return None

        napari_image_class = napari.layers.image.image.Image
        napari_labels_class = napari.layers.labels.labels.Labels
        # If the user has a mask selected, refer to its image layer
        if selected_layer.__class__ == napari_labels_class:
            image_label = self.get_layer_by_name(selected_layer.metadata["associated_image_name"])
        elif selected_layer.__class__ == napari_image_class:
            image_label = selected_layer
        else:
            return None

        if type_of_mask == "axon":
            return self.get_layer_by_name(image_label.metadata["associated_axon_mask_name"])
        elif type_of_mask == "myelin":
            return self.get_layer_by_name(image_label.metadata["associated_myelin_mask_name"])
        return None

    def get_axon_layer(self):
        return self.get_mask_layer("axon")

    def get_myelin_layer(self):
        return self.get_mask_layer("myelin")

    def get_pixel_size_with_prompt(self):
        pixel_size, ok_pressed = QInputDialog.getDouble(self, "Enter the pixel size",
                                                        "Enter the pixel size in micrometers", 0.07, 0, 1000, 10)
        if ok_pressed:
            return pixel_size
        else:
            return None

    def show_info_message(self, message):
        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Information)
        message_box.setText(message)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.exec()

    def show_ok_cancel_message(self, message):
        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Information)
        message_box.setText(message)
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        if message_box.exec() == QMessageBox.Ok:
            return True
        else:
            return False

    def get_logo(self):
        ads_path = Path(AxonDeepSeg.__file__).parents[0]
        logo_file = ads_path / "logo_ads-alpha_small.png"
        logo_label = QLabel(self)
        logo_pixmap = QPixmap(str(logo_file))
        logo_label.setPixmap(logo_pixmap)
        logo_label.resize(logo_pixmap.width(), logo_pixmap.height())
        return logo_label

    def get_citation_string(self):
        """
        This function returns the AxonDeepSeg paper citation.
        :return: The AxonDeepSeg citation
        :rtype: string
        """
        return (
            "If you use this work in your research, please cite it as follows: \n"
            "Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). "
            "AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional "
            "neural networks. Scientific Reports, 8(1), 3816. "
            "Link to paper: https://doi.org/10.1038/s41598-018-22181-4. \n"
            "Copyright (c) 2018 NeuroPoly (Polytechnique Montreal)"
        )


class ApplyModelThread(QtCore.QThread):
    model_applied_signal = Signal()
    def __init__(self):
        super().__init__()
        # Those values must not be None before calling run()
        self.selected_layer = None
        self.image_directory = None
        self.path_testing_image = None
        self.path_model = None
        self.overlap_value = None
        self.zoom_factor = None
        self.no_patch = False
        self.gpu_id = 0
        self.task_finished_successfully = False

    def run(self):
        self.task_finished_successfully = False
        try:
            segment.segment_image(
                path_testing_image=self.path_testing_image,
                path_model=self.path_model,
                overlap_value=self.overlap_value,
                acquired_resolution=self.selected_layer.metadata["pixel_size"],
                zoom_factor=self.zoom_factor,
                gpu_id=self.gpu_id,
                no_patch=self.no_patch,
                verbosity_level=3
            )
            self.task_finished_successfully = True
        except SystemExit as err:
            if err.code == 4:
                self.show_info_message(
                    "Resampled image smaller than model's patch size. Please take a look at your terminal \n"
                    "for the minimum zoom factor value to use (option available in the Settings menu)."
                )
            self.task_finished_successfully = False
        self.model_applied_signal.emit()
