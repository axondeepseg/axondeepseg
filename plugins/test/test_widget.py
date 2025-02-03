# coding: utf-8

import pytest
import napari
from ads_napari._widget import ADSplugin
import numpy as np
from unittest.mock import patch
from pathlib import Path
from ads_base.ads_utils import imread, imwrite
from qtpy.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from vispy.util import keys
import tempfile
import copy
import time

class ImageLoadedEvent(object):
    def __init__(self, data):
        self.value = napari.layers.Image(data)

class TestCore(object):
    def setup_method(self):
        # Get current file folder
        self.current_folder = Path(__file__).parent.resolve()
        self.mask_path = Path(self.current_folder / '../../test/__test_files__/__test_demo_files__/image_seg-axonmyelin.png')
        self.image_path = Path(self.current_folder / '../../test/__test_files__/__test_demo_files__/image.png')
    
        self.known_axon_world_coords = (512.5,249.5) # x,y here are flipped l-r from data coords, as image is flipped l-r when placed in viewer
        self.known_axon_data_coords = (5,-6) # Top right of image

        self.known_myelin_world_coords = (484.8,249.5)
        self.known_myelin_data_coords = (5,-106)

        self.known_background_world_coords = (484.8,249.5)
        self.known_background_data_coords = (5,-106)

        self.known_background_world_coords = (365.3,249.5)
        self.known_background_data_coords = (5,-506)

        self.expected_myelin_metrics_message = "Axon index: 9\nAxon diameter: 6.66 μm\nMyelin thickness: 0.85 μm\ng-ratio: 0.80\nTouches border: True"
    def teardown_method(self):
        pass

    # --------------initial tests-------------- #
    @pytest.mark.integration
    def test_on_layer_added_updates_image_loaded_after_plugin_start(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)

        # Assert initial conditions
        assert wdg.image_loaded_after_plugin_start == False

        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        # Assert that image_loaded_after_plugin_start state changed
        assert wdg.image_loaded_after_plugin_start == True

    @pytest.mark.integration
    def test_on_load_mask_button_click(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)

        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        # Assert that image_loaded_after_plugin_start state changed
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)
        
        # Asserts napari's viewer layers
        assert 'Image_seg-axon' in viewer.layers
        assert 'Image_seg-myelin' in viewer.layers

        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                assert np.all(np.unique(layer.data) == [0,1])

    @pytest.mark.integration
    def test_on_remove_axons_click_user_forgets_to_load_image(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User forgets to loads image
        # Do nothing

        # Assert that image_loaded_after_plugin_start state changed
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)
        
        assert wdg.remove_axon_state == False
        assert wdg.remove_axons_button.isChecked() == False
        
    @pytest.mark.integration
    def test_on_remove_axons_click_with_missing_axonmyelin(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        # Assert that image_loaded_after_plugin_start state changed
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)
        
        assert wdg.remove_axon_state == False
        assert wdg.remove_axons_button.isChecked() == False

    @pytest.mark.integration
    def test_on_remove_axons_click_no_morphometrics_computed(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Assert that image_loaded_after_plugin_start state changed
        # Default state
        assert wdg.remove_axon_state == False
        assert wdg.remove_axons_button.isChecked() == False
        assert wdg.im_axonmyelin_label is None

        ## Simulate Remove Axons button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            # Simulate a button click
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)

        # Expected state
        assert wdg.remove_axon_state == True
        assert wdg.remove_axons_button.isChecked() == True
        assert wdg.im_axonmyelin_label is not None

    @pytest.mark.integration
    def test_on_remove_axons_click_axon_pixel(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Simulate Remove Axons button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            # Simulate a button click
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)

        # Find a pixel in the canvase where axon is 0
        axon_layer = wdg.get_axon_layer()
        myelin_layer = wdg.get_myelin_layer()

        world_coords = self.known_axon_world_coords
        data_coords = self.known_axon_data_coords

        assert axon_layer.data[int(data_coords[0]), int(data_coords[1])] == 1
        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 0
        assert axon_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 0
        assert myelin_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 1

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)

        # Assert that axon label is now 0
        axon_layer.refresh()
        assert axon_layer.data[data_coords[0], data_coords[1]] == 0

        # Also assert that myelin is 0
        myelin_layer.refresh()
        assert myelin_layer.data[data_coords[0], data_coords[1]] == 0

        # Assert that myelin pixel label was also set to 0 for this axon
        assert axon_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 0
        assert myelin_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 0

    @pytest.mark.integration
    def test_on_remove_axons_click_myelin_pixel(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Simulate Remove Axons button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            # Simulate a button click
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)

        # Find a pixel in the canvase where myelin is 1
        axon_layer = wdg.get_axon_layer()
        myelin_layer = wdg.get_myelin_layer()

        world_coords = self.known_myelin_world_coords
        data_coords = self.known_myelin_data_coords

        assert axon_layer.data[int(data_coords[0]), int(data_coords[1])] == 0
        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 1

        assert axon_layer.data[int(self.known_axon_data_coords[0]), int(self.known_axon_data_coords[1])] == 1
        assert myelin_layer.data[int(self.known_axon_data_coords[0]), int(self.known_axon_data_coords[1])] == 0


        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)

        # Assert that axon label is now 0
        axon_layer.refresh()
        assert axon_layer.data[int(data_coords[0]), int(data_coords[0])] == 0

        # Also assert that myelin is 0
        myelin_layer.refresh()
        myelin_layer = wdg.get_myelin_layer()

        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 0

        # Assert that myelin pixel label was also set to 0 for this axon
        assert axon_layer.data[int(self.known_axon_data_coords[0]), int(self.known_axon_data_coords[1])] == 0
        assert myelin_layer.data[int(self.known_axon_data_coords[0]), int(self.known_axon_data_coords[1])] == 0

    @pytest.mark.integration
    def test_remove_axon_undo(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Simulate Remove Axons button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            # Simulate a button click
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)

        # Find a pixel in the canvase where axon is 0
        axon_layer = wdg.get_axon_layer()
        myelin_layer = wdg.get_myelin_layer()

        world_coords = self.known_axon_world_coords
        data_coords = self.known_axon_data_coords

        assert axon_layer.data[int(data_coords[0]), int(data_coords[1])] == 1
        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 0
        assert axon_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 0
        assert myelin_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 1

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)

        # Assert that axon label is now 0
        axon_layer.refresh()
        assert axon_layer.data[data_coords[0], data_coords[1]] == 0

        # Also assert that myelin is 0
        myelin_layer.refresh()
        assert myelin_layer.data[data_coords[0], data_coords[1]] == 0

        # Assert that myelin pixel label was also set to 0 for this axon
        assert axon_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 0
        assert myelin_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 0

        ## Trigger undo (simulate CTRL+Z)
        viewer.layers[axon_layer.name].undo()
        viewer.layers[myelin_layer.name].undo()
        
        # Assert original axon is restored
        assert axon_layer.data[int(data_coords[0]), int(data_coords[1])] == 1
        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 0
        assert axon_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 0
        assert myelin_layer.data[int(self.known_myelin_data_coords[0]), int(self.known_myelin_data_coords[1])] == 1

    @pytest.mark.integration
    def test_on_remove_axons_click_background_pixel(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Simulate Remove Axons button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            # Simulate a button click
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)

        # Find a pixel in the canvase where myelin is 1
        axon_layer = wdg.get_axon_layer()
        myelin_layer = wdg.get_myelin_layer()

        world_coords = self.known_background_world_coords
        data_coords = self.known_background_data_coords

        assert axon_layer.data[int(data_coords[0]), int(data_coords[1])] == 0
        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 0

        original_axon = copy.deepcopy(axon_layer.data)
        original_myelin = copy.deepcopy(myelin_layer.data)

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)

        # Assert label layers unchanged
        axon_layer.refresh()

        myelin_layer.refresh()

        assert np.all(axon_layer.data == original_axon)
        assert np.all(myelin_layer.data == original_myelin)


    @pytest.mark.integration
    def test_on_remove_axons_click_layer_changed_to_label_prior_click(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Simulate Remove Axons button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            # Simulate a button click
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)

        world_coords = self.known_background_world_coords

        ## Change active layer
        viewer.layers.selection = [wdg.get_axon_layer()]

        ## Click that pixel
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)

        # Assert expected message was shown for this pixel
        assert wdg.last_message == "Image layer must be selected."

    @pytest.mark.integration
    def test_on_show_axon_metrics_with_missing_axonmyelin(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        # Assert that image_loaded_after_plugin_start state changed
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            QTest.mouseClick(wdg.show_axon_metrics_button, Qt.LeftButton)
        
        assert wdg.show_axon_metrics_state == False
        assert wdg.show_axon_metrics_button.isChecked() == False

    @pytest.mark.integration
    def test_on_show_axon_metrics_click_no_morphometrics_computed(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None
        assert wdg.stats_dataframe is None

        ## Assert that image_loaded_after_plugin_start state changed
        # Default state
        assert wdg.show_axon_metrics_state == False
        assert wdg.show_axon_metrics_button.isChecked() == False
        assert wdg.im_axonmyelin_label is None

        ## Simulate Show Axon Morphometris button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            with patch("PyQt5.QtWidgets.QInputDialog.getDouble", return_value=(0.07, True)):
                with tempfile.NamedTemporaryFile(prefix='Morphometrics', suffix='.csv', delete=False) as temp_file:
                    with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(temp_file.name, None)):
                        # Simulate a button click
                        QTest.mouseClick(wdg.show_axon_metrics_button, Qt.LeftButton)

        # Expected state
        assert wdg.show_axon_metrics_state == True
        assert wdg.show_axon_metrics_button.isChecked() == True
        assert wdg.im_axonmyelin_label is not None
        assert wdg.stats_dataframe is not None

    @pytest.mark.integration
    def test_on_show_axon_metrics_click_no_morphometrics_computed_user_cancels_pixel(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Assert that image_loaded_after_plugin_start state changed
        # Default state
        assert wdg.show_axon_metrics_state == False
        assert wdg.show_axon_metrics_button.isChecked() == False
        assert wdg.im_axonmyelin_label is None

        ## Simulate Show Axon Morphometris button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            with patch("PyQt5.QtWidgets.QInputDialog.getDouble", return_value=(0.07, False)):
                    # Simulate a button click
                    QTest.mouseClick(wdg.show_axon_metrics_button, Qt.LeftButton)

        # Expected state
        assert wdg.show_axon_metrics_state == False
        assert wdg.show_axon_metrics_button.isChecked() == False
        assert wdg.im_axonmyelin_label is None

    @pytest.mark.integration
    def test_on_show_axon_metrics_click_axon_pixel(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Simulate Show Axon Morphometris button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            with patch("PyQt5.QtWidgets.QInputDialog.getDouble", return_value=(0.07, True)):
                with tempfile.NamedTemporaryFile(prefix='Morphometrics', suffix='.csv', delete=False) as temp_file:
                    with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(temp_file.name, None)):
                        # Simulate a button click
                        QTest.mouseClick(wdg.show_axon_metrics_button, Qt.LeftButton)

        # Find a pixel in the canvase where axon is 1
        axon_layer = wdg.get_axon_layer()
        myelin_layer = wdg.get_myelin_layer()

        world_coords = self.known_axon_world_coords
        data_coords = self.known_axon_data_coords

        assert axon_layer.data[int(data_coords[0]), int(data_coords[1])] == 1
        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 0

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.ALT]), button=0)

        # Assert expected message was shown for this pixel
        assert wdg.last_message == self.expected_myelin_metrics_message

    @pytest.mark.integration
    def test_on_show_axon_metrics_click_background_pixel(self, make_napari_viewer):
        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(imread(self.image_path), rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        ## Simulate Show Axon Morphometris button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            with patch("PyQt5.QtWidgets.QInputDialog.getDouble", return_value=(0.07, True)):
                with tempfile.NamedTemporaryFile(prefix='Morphometrics', suffix='.csv', delete=False) as temp_file:
                    with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(temp_file.name, None)):
                        # Simulate a button click
                        QTest.mouseClick(wdg.show_axon_metrics_button, Qt.LeftButton)

        world_coords = self.known_background_world_coords

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.ALT]), button=0)

        # Assert expected message was shown for this pixel
        assert wdg.last_message == "Backround pixel - no morphometrics to report"

    @pytest.mark.integration
    def test_on_remove_axons_performance_large_image(self,make_napari_viewer):
        # Create a large synthetic int image (e.g., 5000,5000)
        large_image = np.random.randint(0, 256, size=(5000, 5000), dtype=np.uint8)
        large_mask = np.zeros((5000, 5000))
        large_mask[self.known_axon_data_coords]=255

        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(large_image, rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        # Create temp file for axonmyelin mask using large_image data
        with tempfile.NamedTemporaryFile(prefix='large_image_mask', suffix='.png', delete=False) as temp_file:
            imwrite(temp_file.name, large_mask)
            self.mask_path = Path(temp_file.name)

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(temp_file.name), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        # Time the morphometrics computation
        start_time = time.time()

        ## Simulate Remove Axon button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            QTest.mouseClick(wdg.remove_axons_button, Qt.LeftButton)

        world_coords = self.known_axon_world_coords

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)


        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 10.0  # Adjust threshold as needed


    @pytest.mark.integration
    def test_on_show_axon_metrics_performance_large_image(self,make_napari_viewer):
        # Create a large synthetic int image (e.g., 5000,5000)
        large_image = np.random.randint(0, 256, size=(5000, 5000), dtype=np.uint8)
        large_mask = np.zeros((5000, 5000))
        large_mask[self.known_axon_data_coords]=255

        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        viewer.add_image(large_image, rgb=False)
        
        ## User loads image
        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        # Create temp file for axonmyelin mask using large_image data
        with tempfile.NamedTemporaryFile(prefix='large_image_mask', suffix='.png', delete=False) as temp_file:
            imwrite(temp_file.name, large_mask)
            self.mask_path = Path(temp_file.name)

        ## User loads mask
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(temp_file.name), '')):
            with patch('ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)

        ## User omits computing morphometrics via button
        assert wdg.im_axonmyelin_label is None

        # Time the morphometrics computation
        start_time = time.time()

        ## Simulate Show Axon Morphometris button click
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            with patch("PyQt5.QtWidgets.QInputDialog.getDouble", return_value=(0.07, True)):
                with tempfile.NamedTemporaryFile(prefix='Morphometrics', suffix='.csv', delete=False) as temp_file:
                    with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(temp_file.name, None)):
                        # Simulate a button click
                        QTest.mouseClick(wdg.show_axon_metrics_button, Qt.LeftButton)

        world_coords = self.known_axon_world_coords

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.ALT]), button=0)

        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 10.0  # Adjust threshold as needed

    @pytest.mark.integration
    def test_on_show_axon_metrics_warns_user_slow_very_large_image(self,make_napari_viewer):
        # Create a large synthetic int image (e.g., 5001,5000)
        large_image = np.random.randint(0, 256, size=(5001, 5000), dtype=np.uint8)
        large_mask = np.zeros((5001, 500))
        large_mask[self.known_axon_data_coords]=255

        ## User opens plugin
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)
        
        ## User loads image
        with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
            viewer.add_image(large_image, rgb=False)

        wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))

        # Assert expected message was shown 
        assert wdg.last_message == "Large image loaded (greater than 5000 * 5000 pixels) - some plugin features may be slow"