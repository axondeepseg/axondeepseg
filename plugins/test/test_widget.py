# coding: utf-8

import pytest
import napari
from napari_ADS._widget import ADSplugin
import numpy as np
from unittest.mock import patch
from pathlib import Path
from AxonDeepSeg.ads_utils import imread, imwrite
from qtpy.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from vispy.util import keys
import tempfile

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
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)

        # Assert that axon label is now 0
        axon_layer.refresh()
        assert axon_layer.data[data_coords[0], data_coords[1]] == 0

        # Also assert that myelin is 0
        myelin_layer.refresh()
        assert myelin_layer.data[data_coords[0], data_coords[1]] == 0

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
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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

        ## Click that pixel
        viewer.window.qt_viewer.canvas.events.mouse_press(pos=(world_coords[0], world_coords[1]), modifiers=([keys.CONTROL]), button=0)

        # Assert that axon label is now 0
        axon_layer.refresh()
        assert axon_layer.data[int(data_coords[0]), int(data_coords[0])] == 0

        # Also assert that myelin is 0
        myelin_layer.refresh()
        myelin_layer = wdg.get_myelin_layer()

        assert myelin_layer.data[int(data_coords[0]), int(data_coords[1])] == 0

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
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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
            with patch("PyQt5.QtWidgets.QInputDialog.getDouble", return_value=(0.07, True)):
                with tempfile.NamedTemporaryFile(prefix='Morphometrics', suffix='.csv', delete=False) as temp_file:
                    with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(temp_file.name, None)):
                        # Simulate a button click
                        QTest.mouseClick(wdg.show_axon_metrics_button, Qt.LeftButton)

        # Expected state
        assert wdg.show_axon_metrics_state == True
        assert wdg.show_axon_metrics_button.isChecked() == True
        assert wdg.im_axonmyelin_label is not None

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
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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
