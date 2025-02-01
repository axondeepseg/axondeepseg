# coding: utf-8

import pytest
import napari
from napari_ADS._widget import ADSplugin
import numpy as np
from unittest.mock import patch
from pathlib import Path
from AxonDeepSeg.ads_utils import imread
class ImageLoadedEvent(object):
    def __init__(self, data):
        self.value = napari.layers.Image(data)

class TestCore(object):
    def setup_method(self):
        # Get current file folder
        self.current_folder = Path(__file__).parent.resolve()
        self.mask_path = Path(self.current_folder / '../../test/__test_files__/__test_demo_files__/image_seg-axonmyelin.png')
        self.image_path = Path(self.current_folder / '../../test/__test_files__/__test_demo_files__/image.png')
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
        print(napari.layers.labels.labels.Labels)

        # Assert that image_loaded_after_plugin_start state changed
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(self.mask_path), '')):
            with patch('napari_ADS._widget.ADSplugin.show_ok_cancel_message', return_value=(False, '')):
                wdg._on_load_mask_button_click()
        
        # Print napari's viewer layers
        assert 'Image_seg-axon' in viewer.layers
        assert 'Image_seg-myelin' in viewer.layers

        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                print(np.unique(layer.data))
                assert np.all(np.unique(layer.data) == [0,1])
