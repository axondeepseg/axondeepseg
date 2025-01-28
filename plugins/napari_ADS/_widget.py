from typing import TYPE_CHECKING

import os, sys
from pathlib import Path

import AxonDeepSeg
import AxonDeepSeg.params as config
import numpy as np
import qtpy.QtCore
from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QWidget,
    QComboBox,
    QFileDialog,
    QLabel,
    QPlainTextEdit,
    QInputDialog,
    QMessageBox,
)
from qtpy.QtCore import QStringListModel, QObject, Signal
from qtpy.QtGui import QPixmap

from AxonDeepSeg import ads_utils, segment, postprocessing, params
import AxonDeepSeg.morphometrics.compute_morphometrics as compute_morphs
from AxonDeepSeg.params import axonmyelin_suffix, axon_suffix, myelin_suffix

import napari
from napari.utils.notifications import show_info
from .settings_menu_ui import Ui_Settings_menu_ui
from vispy.util import keys

_SHIFT = keys.SHIFT,
_CONTROL =  keys.CONTROL
_ALT = keys.ALT

class ADSsettings:
    """Plugin settings class.

    This class handles everything related to the parameters used in the ADS plugin, including the frame for the settings
    menu.
    """

    def __init__(self, ads_plugin):
        """Constructor for the ADSsettings class.

        Args:
            ads_plugin: An instance of the ADS plugin that uses the ADSsettings class.

        Returns:
            None
        """
        self.ads_plugin = ads_plugin

        # Declare the settings used
        self.axon_shape = "circle"
        self._axon_shape_selection_index = 0
        self.gpu_id = -1
        self.n_gpus = ads_utils.check_available_gpus(None)
        self.max_gpu_id = self.n_gpus - 1 if self.n_gpus > 0 else 0
        self.setup_settings_menu()

    def setup_settings_menu(self):
        """Sets up the settings menu for the AxonDeepSeg plugin.

        The settings menu contains options for the user to adjust certain parameters for the segmentation process.

        Returns:
            None
        """
        self.Settings_menu_ui = QtWidgets.QDialog(self.ads_plugin)
        self.ui = Ui_Settings_menu_ui()
        self.ui.setupUi(self.Settings_menu_ui)
        self.ui.done_button.clicked.connect(self._on_done_button_click)

        self.ui.axon_shape_comboBox.currentIndexChanged.connect(
            self._on_axon_shape_changed
        )
        self.ui.gpu_id_spinBox.valueChanged.connect(self._on_gpu_id_changed)

    def create_settings_menu(self):
        """Creates the settings menu and sets the values of its UI elements to the current settings.

        Sets the values in the settings menu of the overlap value, zoom factor, axon shape, no patch,
        and GPU ID to the current values of the corresponding settings. Shows the settings menu.

        Returns:
            None
        """
        self.ui.axon_shape_comboBox.setCurrentIndex(
            self._axon_shape_selection_index
        )
        self.ui.gpu_id_spinBox.setValue(self.gpu_id)
        self.ui.gpu_id_spinBox.setMaximum(self.max_gpu_id)
        self.Settings_menu_ui.show()

    def _on_done_button_click(self):
        """Closes the settings menu dialog when the user clicks the 'Done' button.

        Returns:
            None
        """
        self.Settings_menu_ui.close()

    def _on_axon_shape_changed(self):
        """Update the axon shape attribute with the value from the UI's axon shape combo box.

        This method is called when the axon shape value combo box value is changed in the UI. It retrieves the new value
        from the combo box and updates the axon_shape attribute of the class instance.

        Returns:
            None
        """
        self.axon_shape = self.ui.axon_shape_comboBox.currentText()
        self._axon_shape_selection_index = (
            self.ui.axon_shape_comboBox.currentIndex()
        )

    def _on_gpu_id_changed(self):
        """Update the GPU ID value attribute with the value from the UI's GPU ID spin box.

        This method is called when the GPU ID value spin box value is changed in the UI. It retrieves the new value
        from the spin box and updates the gpu_id attribute of the class instance.

        Returns:
            None
        """
        self.gpu_id = self.ui.gpu_id_spinBox.value()


class ADSplugin(QWidget):
    """Plugin class.

    This class handles the ADS plugin widget.
    """

    def __init__(self, napari_viewer):
        """Constructor for the ADS plugin widget.

        This method initializes the ADS plugin widget. It sets the viewer object as an attribute of the class and
        initializes the ADSsettings object as an attribute of the widget.

        The method also creates the user interface elements for the ADS plugin widget.

        Args:
            param napari_viewer: The napari viewer object.

        Returns:
            None
        """
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
            '<a href="https://axondeepseg.readthedocs.io/en/latest/">Need help? Read the documentation</a>'
        )

        self.available_models = ads_utils.get_existing_models_list()
        self.model_selection_combobox = QComboBox()
        self.model_selection_combobox.addItems(
            ["Select the model"] + self.available_models
        )

        self.apply_model_button = QPushButton("Apply ADS model")
        self.apply_model_button.clicked.connect(
            self._on_apply_model_button_click
        )
        self.apply_model_thread = ApplyModelThread()
        self.apply_model_thread.model_applied_signal.connect(
            self._on_model_finished_apply
        )

        load_mask_button = QPushButton("Load mask")
        load_mask_button.clicked.connect(self._on_load_mask_button_click)

        fill_axons_button = QPushButton("Fill axons")
        fill_axons_button.clicked.connect(self._on_fill_axons_click)

        save_segmentation_button = QPushButton("Save segmentation")
        save_segmentation_button.clicked.connect(
            self._on_save_segmentation_button
        )

        compute_morphometrics_button = QPushButton("Compute morphometrics")
        compute_morphometrics_button.clicked.connect(
            self._on_compute_morphometrics_button
        )

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

        # Connect the mouse click event to the handler
        self.viewer.layers.events.inserted.connect(self._on_layer_added)

    def _on_layer_added(self, event):
        """Handler for when a layer is added to the viewer.

        Args:
            event: The event object containing the layer that was added.

        Returns:
            None
        """
        layer = event.value
        if isinstance(layer, napari.layers.Image):
            layer.mouse_drag_callbacks.append(self._on_image_click)

    def _on_image_click(self, layer, event):
        """Handler for when an image layer is clicked.

        Args:
            layer: The image layer that was clicked.
            event: The event object containing the click position and keyboard modifiers.

        Returns:
            None
        """
        if _CONTROL in event.modifiers:  # Command key on macOS
            data_coordinates = layer.world_to_data(event.position)
            cords = np.round(data_coordinates).astype(int)
            
            # Ensure the coordinates are within the bounds of the image
            if 0 <= cords[1] < self.im_instance_seg.shape[1] and 0 <= cords[0] < self.im_instance_seg.shape[0]:
                # Get the RGB value at the clicked position
                rgb_value = self.im_instance_seg[cords[0], cords[1]]
                
                # If you need to convert the RGB value to a unique identifier (e.g., for segmentation)
                # You can hash the RGB tuple to a single value (e.g., using a tuple as a dictionary key)
                axon_num = tuple(rgb_value)  # Use the RGB tuple as the identifier
                
                print("Image instance segmentation array:")
                print(self.im_instance_seg)
                print("Image shape:", self.im_instance_seg.shape)
                print(f"RGB value at clicked position: {rgb_value}")
                print(f"Axon identifier (RGB tuple): {axon_num}")

                # Get the indices for each region with the same RGB value
                idx = np.where((self.im_instance_seg == rgb_value).all(axis=-1))
                print(f"Shape of idx tuple: {len(idx)}")
                print(f"Indices of axon {axon_num}: {idx}")
                
                show_info(f"Clicked at {cords} with Command key pressed, on axon {axon_num}")


                axon_layer = self.get_axon_layer()
                myelin_layer = self.get_myelin_layer()

                if (axon_layer is None) or (myelin_layer is None):
                    self.show_info_message("One or more masks missing")
                    return

                
                axon_layer._save_history(
                    (
                        idx,
                        np.array(axon_layer.data[idx], copy=True),
                        0,
                    )
                )
                axon_layer.data[idx] = 0
                axon_layer.refresh()

                myelin_layer._save_history(
                    (
                        idx,
                        np.array(myelin_layer.data[idx], copy=True),
                        0,
                    )
                )
                myelin_layer.data[idx] = 0
                myelin_layer.refresh()
            else:
                show_info("Clicked position is out of bounds.")
        elif _ALT in event.modifiers:  # Option key on macOS
            data_coordinates = layer.world_to_data(event.position)
            cords = np.round(data_coordinates).astype(int)
            show_info(f"Clicked at {cords} with Alt key pressed")
        else:
            data_coordinates = layer.world_to_data(event.position)
            cords = np.round(data_coordinates).astype(int)
            show_info(f"Clicked at {cords}")
            return



    def try_to_get_pixel_size_of_layer(self, layer):
        """Method to attempt to retrieve the pixel size of an image layer.
        This method attempts to retrieve the pixel size of the image represented by the layer passed as input parameter.
        It will return the value found in "pixel_size_in_micrometers.txt" if the file exists.

        Args:
            layer: The napari image layer for which to try to retrieve the pixel size.

        Returns:
            - The pixel size in um (float) if it found it.
            - None if it didn't find the pixel size
        """
        image_path = Path(layer.source.path)
        image_directory = image_path.parents[0]

        # Check if the pixel size txt file exist in the image_directory
        pixel_size_exists = (
            image_directory / "pixel_size_in_micrometer.txt"
        ).exists()

        if pixel_size_exists:
            resolution_file = open(
                str((image_directory / "pixel_size_in_micrometer.txt")), "r"
            )
            pixel_size_float = float(resolution_file.read())
            return pixel_size_float
        else:
            return None

    def add_layer_pixel_size_to_metadata(self, layer):
        """Method to add the pixel size of an image layer to its metadata.

        This method attempts to retrieve the pixel size of the input image layer using the
        'try_to_get_pixel_size_of_layer' method.

        Args:
            layer: The napari image layer for which to add the pixel size metadata.

        Returns:
            bool:
                True, if the pixel size metadata was successfully added to the layer.
                False, if the pixel size metadata could not be retrieved or added to the layer.
        """
        pixel_size = self.try_to_get_pixel_size_of_layer(layer)
        if pixel_size is not None:
            layer.metadata["pixel_size"] = pixel_size
            return True
        else:
            return False

    def _on_apply_model_button_click(self):
        """Apply the selected AxonDeepSeg model to the active layer of the viewer.

        Returns:
            None
        """
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

        self.apply_model_button.setEnabled(False)
        self.apply_model_thread.selected_layer = selected_layer
        self.apply_model_thread.image_directory = image_directory
        self.apply_model_thread.path_image = Path(
            selected_layer.source.path
        )
        self.apply_model_thread.path_model = model_path
        self.apply_model_thread.gpu_id = self.settings.gpu_id
        show_info(
            "Applying ADS model... This can take a few seconds. Check the console for more information."
        )
        self.apply_model_thread.start()

    def _on_model_finished_apply(self):
        """Callback function called when the apply model thread finishes.

        This method gets the results of the apply model thread and updates the viewer and layer
        metadata with the axon and myelin masks generated by the model. If the thread
        finished successfully, the method gets the path and name of the axon and myelin masks
        generated by the model, reads the masks from disk, and adds them to the viewer as separate
        labels.

        Returns:
            None
        """

        self.apply_model_button.setEnabled(True)
        if not self.apply_model_thread.task_finished_successfully:
            self.show_info_message(
                "Couldn't apply the ADS model. Please check the console or terminal for more information."
            )
            return

        selected_layer = self.apply_model_thread.selected_layer
        image_directory = self.apply_model_thread.image_directory
        image_name_no_extension = selected_layer.name
        axon_mask_path = image_directory / (
            image_name_no_extension + str(axon_suffix)
        )
        axon_mask_name = image_name_no_extension + axon_suffix.stem
        myelin_mask_path = image_directory / (
            image_name_no_extension + str(myelin_suffix)
        )
        myelin_mask_name = image_name_no_extension + myelin_suffix.stem

        axon_data = ads_utils.imread(axon_mask_path).astype(bool)
        self.viewer.add_labels(
            axon_data,
            colormap={None: 'transparent', 1: "blue"},
            name=axon_mask_name,
            metadata={"associated_image_name": image_name_no_extension},
        )
        myelin_data = ads_utils.imread(myelin_mask_path).astype(bool)
        self.viewer.add_labels(
            myelin_data,
            colormap={None: 'transparent', 1: "red"},
            name=myelin_mask_name,
            metadata={"associated_image_name": image_name_no_extension},
        )
        selected_layer.metadata["associated_axon_mask_name"] = axon_mask_name
        selected_layer.metadata[
            "associated_myelin_mask_name"
        ] = myelin_mask_name

    def _on_load_mask_button_click(self):
        """Handles the click event of the 'Load Mask' button.

        The method loads a mask file selected by the user and creates two new labels to represent
        the Axon and Myelin masks. The masks are associated with the currently selected microscopy
        image, and metadata is added to the image layer to keep a link between them.

        Returns:
            None
        """
        microscopy_image_layer = self.get_microscopy_image()
        if microscopy_image_layer is None:
            self.show_info_message("No single image selected/detected")
            return

        mask_file_path, _ = QFileDialog.getOpenFileName(
            self, "Select the mask you wish to load"
        )
        if mask_file_path == "":
            return

        if not self.show_ok_cancel_message(
            "The mask will be associated with " + microscopy_image_layer.name
        ):
            return

        img_png2D = ads_utils.imread(mask_file_path)
        # Extract the Axon mask
        axon_data = img_png2D > 200
        axon_data = axon_data.astype(np.uint8)
        axon_mask_name = microscopy_image_layer.name + config.axon_suffix.stem
        # Extract the Myelin mask
        myelin_data = (img_png2D > 100) & (img_png2D < 200)
        myelin_data = myelin_data.astype(np.uint8)
        myelin_mask_name = (
            microscopy_image_layer.name + config.myelin_suffix.stem
        )

        # Load the masks and add metadata to the files to keep a link between them
        self.viewer.add_labels(
            axon_data,
            colormap={ None: 'transparent', 1: "blue"},
            name=axon_mask_name,
            metadata={"associated_image_name": microscopy_image_layer.name},
        )
        self.viewer.add_labels(
            myelin_data,
            colormap={ None: 'transparent', 1: "red"},
            name=myelin_mask_name,
            metadata={"associated_image_name": microscopy_image_layer.name},
        )
        microscopy_image_layer.metadata[
            "associated_axon_mask_name"
        ] = axon_mask_name
        microscopy_image_layer.metadata[
            "associated_myelin_mask_name"
        ] = myelin_mask_name

    def _on_fill_axons_click(self):
        """Handles the click event of the 'Fill Axons' button.

        The method fills the holes in the myelin mask and extracts the axons from it. It then sets the
        corresponding values in the axon layer to 1, effectively updating the axon mask.

        Returns:
            None
        """
        axon_layer = self.get_axon_layer()
        myelin_layer = self.get_myelin_layer()

        if (axon_layer is None) or (myelin_layer is None):
            self.show_info_message("One or more masks missing")
            return

        myelin_array = np.array(myelin_layer.data, copy=True)
        axon_extracted_array = postprocessing.fill_myelin_holes(myelin_array)
        axon_array_indexes = np.where(axon_extracted_array > 0)
        axon_layer._save_history(
            (
                axon_array_indexes,
                np.array(axon_layer.data[axon_array_indexes], copy=True),
                1,
            )
        )
        axon_layer.data[axon_array_indexes] = 1
        axon_layer.refresh()

    def _on_save_segmentation_button(self):
        """Handles the click event of the 'Save Segmentation' button.

        The method prompts the user to select a directory where the segmentation images will be saved.
        It then scales the pixel values of the myelin and axon layers to 8-bits and combines them into a single
        image, which is saved as a PNG file. Additionally, the method saves the myelin and axon masks as
        separate PNG files in the same directory.

        Returns:
            None
        """
        axon_layer = self.get_axon_layer()
        myelin_layer = self.get_myelin_layer()

        if (axon_layer is None) or (myelin_layer is None):
            self.show_info_message("One or more masks missing")
            return
        save_path = QFileDialog.getExistingDirectory(
            self, "Select where the segmentation should be saved"
        )
        save_path = Path(save_path)

        # Scale the pixel values of the masks to 255 for image saving
        myelin_array = myelin_layer.data * params.intensity["binary"]
        axon_array = axon_layer.data * params.intensity["binary"]

        myelin_and_axon_array = (myelin_array // 2 + axon_array).astype(
            np.uint8
        )

        microscopy_image_name = axon_layer.metadata["associated_image_name"]
        axon_image_name = microscopy_image_name + str(config.axon_suffix)
        myelin_image_name = microscopy_image_name + str(config.myelin_suffix)
        axonmyelin_image_name = microscopy_image_name + str(
            config.axonmyelin_suffix
        )

        ads_utils.imwrite(
            filename=save_path / axonmyelin_image_name,
            img=myelin_and_axon_array,
        )
        ads_utils.imwrite(
            filename=save_path / myelin_image_name, img=myelin_array
        )
        ads_utils.imwrite(filename=save_path / axon_image_name, img=axon_array)

    def _on_compute_morphometrics_button(self):
        """Compute and save morphometrics statistics for the axon and myelin layers in the viewer.

        Retrieves the axon layer, myelin layer, and microscopy image layer from the viewer.

        If the pixel size of the microscopy image is not already set in the metadata, prompts the user to enter the pixel
        size. Then, prompts the user to select where to save the morphometrics statistics file.

        Computes the axon morphometrics using the axon and myelin data and the pixel size, using the axon shape specified
        in the settings. The resulting statistics dataframe is saved to the selected file location.

        Finally, adds an image to the viewer showing the index image array.

        Returns:
            None
        """
        axon_layer = self.get_axon_layer()
        myelin_layer = self.get_myelin_layer()
        microscopy_image_layer = self.get_microscopy_image()

        if (
            (axon_layer is None)
            or (myelin_layer is None)
            or (microscopy_image_layer is None)
        ):
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
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            caption="Select where to save morphometrics",
            directory=str(default_name),
            filter="CSV file(*.csv)",
        )
        if file_name == "":
            return

        # Compute statistics
        (
            stats_dataframe,
            index_image_array,
            im_instance_seg,
        ) = compute_morphs.get_axon_morphometrics(
            im_axon=axon_data,
            im_myelin=myelin_data,
            pixel_size=pixel_size,
            axon_shape=self.settings.axon_shape,
            return_index_image=True,
            return_instance_seg=True,
        )
        try:
            compute_morphs.save_axon_morphometrics(file_name, stats_dataframe)

        except IOError:
            self.show_info_message("Cannot save morphometrics")

        self.viewer.add_image(
            data=index_image_array,
            rgb=False,
            colormap="yellow",
            blending="additive",
            name="numbers",
        )


        self.stats_dataframe = stats_dataframe
        self.index_image_array = index_image_array
        self.im_instance_seg = im_instance_seg

    def _on_settings_menu_clicked(self):
        """Create and display the settings menu when the settings menu button is clicked.

        Returns:
            None
        """
        self.settings.create_settings_menu()

    def get_layer_by_name(self, name_of_layer):
        """Retrieve the layer with the specified name from the viewer.

        Args:
            name_of_layer: The name of the layer to retrieve.

        Returns:
            The layer object with the specified name, or `None` if no such layer exists in the viewer.
        """
        for layer in self.viewer.layers:
            if layer.name == name_of_layer:
                return layer

    def get_microscopy_image(self):
        """Retrieve the currently selected microscopy image layer from the Napari viewer.

        The layer is retrieved either directly (if it's an Image layer), or through associated metadata (if
        it's a Label layer, i.e. myelin or axon).

        Returns:
            The layer representing the microscopy image.
        """
        selected_layers = self.viewer.layers.selection
        if len(selected_layers) == 0:
            return None
        selected_layer = selected_layers.active
        if selected_layer is None:
            return None

        if selected_layer.__class__ == napari.layers.image.image.Image:
            return selected_layer
        elif selected_layer.__class__ == napari.layers.labels.labels.Labels:
            return self.get_layer_by_name(
                selected_layer.metadata["associated_image_name"]
            )
        else:
            return None

    def get_mask_layer(self, type_of_mask):
        """Return the mask layer of the given type associated with the currently selected image layer.

        Args:
            type_of_mask (str): The type of mask to retrieve. Valid values are 'axon' and 'myelin'.

        Returns:
            Napari layer or None: The mask layer associated with the selected image layer and the
            specified type of mask, or None if no valid mask is found.
        """
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
            image_label = self.get_layer_by_name(
                selected_layer.metadata["associated_image_name"]
            )
        elif selected_layer.__class__ == napari_image_class:
            image_label = selected_layer
        else:
            return None

        if type_of_mask == "axon":
            return self.get_layer_by_name(
                image_label.metadata["associated_axon_mask_name"]
            )
        elif type_of_mask == "myelin":
            return self.get_layer_by_name(
                image_label.metadata["associated_myelin_mask_name"]
            )
        return None

    def get_axon_layer(self):
        """Return the axon mask layer associated with the currently selected image layer.

        Returns:
            Napari layer or None: The axon mask layer associated with the selected image layer, or
            None if no valid axon mask is found.
        """
        return self.get_mask_layer("axon")

    def get_myelin_layer(self):
        """Return the myelin mask layer associated with the currently selected image layer.

        Returns:
            Napari layer or None: The axon mask layer associated with the selected image layer, or
            None if no valid axon mask is found.
        """
        return self.get_mask_layer("myelin")

    def get_pixel_size_with_prompt(self):
        """Displays a dialog box to prompt the user to enter the pixel size in micrometers.

        Returns:
            float or None:
                The entered pixel size in micrometers as a float, or None if the user cancelled the dialog box.
        """
        pixel_size, ok_pressed = QInputDialog.getDouble(
            self,
            "Enter the pixel size",
            "Enter the pixel size in micrometers",
            0.07,
            0,
            1000,
            10,
        )
        if ok_pressed:
            return pixel_size
        else:
            return None

    def show_info_message(self, message):
        """Opens a message box dialog with an informational icon, a message text, and an "Ok" button.

        Args:
            message (str): The text to display in the message box.

        Returns:
            None
        """
        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Information)
        message_box.setText(message)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.exec()

    def show_ok_cancel_message(self, message):
        """Displays a message box with an Ok and Cancel button and prompts the user to confirm an action.

        Args:
            message: The message to display in the message box.

        Returns:
            bool: True if the Ok button is clicked, False if the Cancel button is clicked.
        """
        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Information)
        message_box.setText(message)
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        if message_box.exec() == QMessageBox.Ok:
            return True
        else:
            return False

    def get_logo(self):
        """Return a QLabel object with the AxonDeepSeg logo as its pixmap.

        Returns:
            QLabel: A QLabel object with the AxonDeepSeg logo as its pixmap.
        """
        ads_path = Path(AxonDeepSeg.__file__).parents[0]
        logo_file = ads_path / "logo_ads-alpha_small.png"
        logo_label = QLabel(self)
        logo_pixmap = QPixmap(str(logo_file))
        logo_label.setPixmap(logo_pixmap)
        logo_label.resize(logo_pixmap.width(), logo_pixmap.height())
        return logo_label

    def get_citation_string(self):
        """This function returns the AxonDeepSeg paper citation.

        Returns:
            The AxonDeepSeg citation
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
    """QThread class used to segment an image by applying a model in a separate thread.

    Returns:
        None
    """

    model_applied_signal = Signal()

    def __init__(self):
        """Initializes an instance of the class with default values for attributes.
        Note: their value must be changed to an appropriate value before applying the model.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        # Those values must not be None before calling run()
        self.selected_layer = None
        self.image_directory = None
        self.path_image = None
        self.path_model = None
        self.gpu_id = -1
        self.task_finished_successfully = False

    def run(self):
        """Executes the segmentation process in a separate thread on the selected image layer using the AxonDeepSeg
        model.

        Returns:
            None
        """
        self.task_finished_successfully = False
        try:
            segment.segment_images(
                path_images=[self.path_image],
                path_model=self.path_model,
                gpu_id=self.gpu_id,
                verbosity_level=3,
            )
            self.task_finished_successfully = True
        except SystemExit as err:
            if err.code == 4:
                print(
                    "Resampled image smaller than model's patch size. Please take a look at the lines above \n"
                    "for the minimum zoom factor value to use (option available in the Settings menu)."
                )
            self.task_finished_successfully = False
        self.model_applied_signal.emit()
