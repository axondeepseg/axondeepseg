# coding: utf-8

import pytest
import napari
from AxonDeepSeg.ads_napari._widget import ADSplugin
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
import copy
import time
import sys
import shutil

class ImageLoadedEvent(object):
    def __init__(self, data):
        self.value = napari.layers.Image(data)

class TestCore(object):
    def setup_method(self):
        # Get current file folder
        self.current_folder = Path(__file__).parent.resolve()
        self.mask_path = Path(self.current_folder / '../../test/__test_files__/__test_demo_files__/image_seg-axonmyelin.png')
        self.image_path = Path(self.current_folder / '../../test/__test_files__/__test_demo_files__/image.png')
    
        self.rgb_tmp_dir = Path(self.current_folder / '../../test/__test_files__/__test_color_files__/tmp')
        self.rgb_tmp_dir.mkdir(exist_ok=True)
        
        self.rgb_image_path = Path(self.current_folder / '../../test/__test_files__/__test_color_files__/image_8bit.png')
        
        shutil.copy(self.rgb_image_path, self.rgb_tmp_dir)
        self.rgb_image_tmp_path = self.rgb_tmp_dir / 'image_8bit.png'

        self.expected_myelin_metrics_message = "Axon index: 9\nAxon diameter: 6.66 μm\nMyelin thickness: 0.85 μm\ng-ratio: 0.80\nTouches border: True"
    def teardown_method(self):
        if self.rgb_tmp_dir.exists():
            shutil.rmtree(self.rgb_tmp_dir)
        else:
            pass


    # ------------------User Workflow tests------------------ #
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_rgb_to_greyscale_user_workflow(self, make_napari_viewer, qtbot):
        try:
            ## User opens plugin
            viewer = make_napari_viewer(show=False)
            wdg = ADSplugin(viewer)
            viewer.open(self.rgb_image_tmp_path)

            # Print loaded images
            print(viewer.layers)

            # Select the first layer (index 0)
            selected_layer = viewer.layers[0]
            print(selected_layer.name)  # Print the name of the selected layer

            # Select the first model (index 0 is the text to tell the user to select a model)
            wdg.model_selection_combobox.setCurrentIndex(1)
        
            # User clicks apply model
            with qtbot.waitSignal(wdg.apply_model_thread.model_applied_signal, timeout=300000):
                with patch('AxonDeepSeg.ads_napari._widget.ADSplugin.show_info_message', return_value=(False, '')):
                    wdg.apply_model_button.click()

            # Check that the output images exist
            assert self.rgb_tmp_dir.exists()
            assert any(f.name.endswith('-axon.png') for f in self.rgb_tmp_dir.iterdir())
            assert any(f.name.endswith('-myelin.png') for f in self.rgb_tmp_dir.iterdir())
            assert any(f.name.endswith('-axonmyelin.png') for f in self.rgb_tmp_dir.iterdir())

            # Check that they were made using the greyscale converted images
            for f in self.rgb_tmp_dir.iterdir():
                if f.name.endswith('-axon.png'):
                    assert '_grayscale' in f.name
                if f.name.endswith('-myelin.png'):
                    assert '_grayscale' in f.name
                if f.name.endswith('-axonmyelin.png'):  
                    assert '_grayscale' in f.name

        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails

    # --------------initial tests-------------- #
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_layer_added_updates_image_loaded_after_plugin_start(self, make_napari_viewer):
        try:
            ## User opens plugin
            viewer = make_napari_viewer(show=False)
            wdg = ADSplugin(viewer)
    
            # Assert initial conditions
            assert wdg.image_loaded_after_plugin_start == False
    
            ## User loads image
            wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))
    
            # Assert that image_loaded_after_plugin_start state changed
            assert wdg.image_loaded_after_plugin_start == True
        
        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails

    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_load_mask_button_click(self, make_napari_viewer):
        try:
            ## User opens plugin
            viewer = make_napari_viewer(show=False)
            wdg = ADSplugin(viewer)
            viewer.add_image(imread(self.image_path), rgb=False)
    
            ## User loads image
            wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))
    
            # Assert that image_loaded_after_plugin_start state changed
            with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
                with patch('AxonDeepSeg.ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                    QTest.mouseClick(wdg.load_mask_button, Qt.LeftButton)
            
            # Asserts napari's viewer layers
            assert 'Image_seg-axon' in viewer.layers
            assert 'Image_seg-myelin' in viewer.layers
    
            for layer in viewer.layers:
                if isinstance(layer, napari.layers.Labels):
                    assert np.all(np.unique(layer.data) == [0,1])

        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
    
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_remove_axons_click_user_forgets_to_load_image(self, make_napari_viewer):
        try:
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

        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
    
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_remove_axons_click_with_missing_axonmyelin(self, make_napari_viewer):
        try:
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
        
        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
    
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_remove_axons_click_no_morphometrics_computed(self, make_napari_viewer):
        try:
            ## User opens plugin
            viewer = make_napari_viewer(show=False)
            wdg = ADSplugin(viewer)
            viewer.add_image(imread(self.image_path), rgb=False)
            
            ## User loads image
            wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))
    
            ## User loads mask
            with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
                with patch('AxonDeepSeg.ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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
        
        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
    
    
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_show_axon_metrics_with_missing_axonmyelin(self, make_napari_viewer):
        try: 
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

        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
    
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_show_axon_metrics_click_no_morphometrics_computed(self, make_napari_viewer):
        try:
            ## User opens plugin
            viewer = make_napari_viewer(show=False)
            wdg = ADSplugin(viewer)
            viewer.add_image(imread(self.image_path), rgb=False)
            
            ## User loads image
            wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))
    
            ## User loads mask
            with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
                with patch('AxonDeepSeg.ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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

        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
    
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_show_axon_metrics_click_no_morphometrics_computed_user_cancels_pixel(self, make_napari_viewer):
        try:
        
            ## User opens plugin
            viewer = make_napari_viewer(show=False)
            wdg = ADSplugin(viewer)
            viewer.add_image(imread(self.image_path), rgb=False)
            
            ## User loads image
            wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))
    
            ## User loads mask
            with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
                with patch('AxonDeepSeg.ads_napari._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
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

        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
    
    
    @pytest.mark.skipif(sys.platform == 'linux', reason="Can't test GUI on Linux")
    @pytest.mark.integration
    def test_on_show_axon_metrics_warns_user_slow_very_large_image(self,make_napari_viewer):
        try:
            # Create a large synthetic int image (e.g., 5001,5000)
            large_image = np.random.randint(0, 256, size=(5001, 5000), dtype=np.uint8)
            large_mask = np.zeros((5001, 500))
            large_mask[0,0]=255
    
            ## User opens plugin
            viewer = make_napari_viewer(show=False)
            wdg = ADSplugin(viewer)
            
            ## User loads image
            with patch("PyQt5.QtWidgets.QMessageBox.exec", return_value=QMessageBox.Ok):
                viewer.add_image(large_image, rgb=False)
    
            wdg._on_layer_added(ImageLoadedEvent(imread(self.image_path)))
    
            # Assert expected message was shown 
            assert wdg.last_message == "Large image loaded (greater than 5000 * 5000 pixels) - some plugin features may be slow"
        finally:
            # Clean up layers to prevent OpenGL context issues during teardown
            try:
                viewer.layers.clear()
                # Give some time for OpenGL operations to complete
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Continue with teardown even if cleanup fails
