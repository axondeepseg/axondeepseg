# coding: utf-8

import pytest
import napari
from napari_ADS._widget import ADSplugin
import numpy as np

class ImageLoadedEvent(object):
    value = napari.layers.Image(np.array([[1,2],[3,4]]))

class TestCore(object):
    def setup_method(self):
        pass
    def teardown_method(self):
        pass

    # --------------initial tests-------------- #
    @pytest.mark.unit
    def test_on_layer_added_updates_image_loaded_after_plugin_start(self, make_napari_viewer):
        viewer = make_napari_viewer(show=False)
        wdg = ADSplugin(viewer)

        # Check initial conditions
        assert wdg.image_loaded_after_plugin_start == False
        a = ImageLoadedEvent()
        print(a)
        print(a.value)
        print(isinstance(a.value, napari.layers.Image))
        # Call _on_layer_added
        wdg._on_layer_added(ImageLoadedEvent())

        # Check that image_loaded_after_plugin_start state changed
        assert wdg.image_loaded_after_plugin_start == True
