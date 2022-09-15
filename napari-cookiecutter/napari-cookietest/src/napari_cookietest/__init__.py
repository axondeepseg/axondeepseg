__version__ = "0.0.1"

from ._reader import napari_get_reader
from ._widget import ExampleQWidget, example_magic_widget
from ._writer import write_multiple, write_single_image

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "ExampleQWidget",
    "example_magic_widget",
)
