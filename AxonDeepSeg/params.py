import numpy as np
from pathlib import Path

# segmentation files suffix names
axonmyelin_suffix = Path('_seg-axonmyelin.png')         # axon + myelin segmentation suffix file name
axon_suffix = Path('_seg-axon.png')                     # axon segmentation suffix file name
myelin_suffix = Path('_seg-myelin.png')                 # myelin segmentation suffix file name
index_suffix = Path('_index.png')                       # image with the index of the axons
axonmyelin_index_suffix = Path('_axonmyelin_index.png') # Colored axonmyelin segmentation + the index image
unmyelinated_suffix = Path('_seg-uaxon.png')            # unmyelinated axon segmentation suffix file name
unmyelinated_index_suffix = Path('_uaxon_index.png')    # Colored unmyelinated axon segmentation + the index image
nnunet_suffix=Path('_seg-nnunet.png')                   # nnunet raw segmentation suffix

side_effect_suffixes = tuple(
    [
        str(s) for s in [
            axonmyelin_suffix, axon_suffix, myelin_suffix, index_suffix, 
            axonmyelin_index_suffix, unmyelinated_suffix, unmyelinated_index_suffix,
            nnunet_suffix
        ]
    ]
)

# morphometrics file suffix name
morph_suffix = Path('axon_morphometrics.xlsx')
morph_agg_suffix = Path('subject_morphometrics.xlsx')
unmyelinated_morph_suffix = Path('uaxon_morphometrics.xlsx')
instance_suffix = Path('_instance-map.png')             # Colored instance map of the segmentation

# aggregate morphometrics file suffix name
agg_dir = Path('morphometrics_agg')

# morphometrics statistics analysis
binned_statistics_filename = 'statistics_per_axon_caliber.xlsx'
axon_count_filename = 'axon_count.png'
metrics_names = {
    ('Axon Diameter', 'axon_diam (um)'),
    ('G-Ratio', 'gratio'),
    ('Myelin Thickness', 'myelin_thickness (um)'),
}

# List of valid image extensions
valid_extensions = [
    ".ome.tif",
    ".ome.tiff",
    ".ome.tf2",
    ".ome.tf8",
    ".ome.btf",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg"
    ]

intensity = {
    'binary': np.iinfo(np.uint8).max,
    'axon': np.iinfo(np.uint8).max,
    'myelin': np.iinfo(np.uint8).max // 2,
    'background': np.iinfo(np.uint8).min
}


class Morphometrics_Column_Name:
    """
    A Morphometrics_Column_Name indicates the key_name and display_name of a morphometrics dataframe column
    :param key_name: The column name used in the dataframe variable
    :type key_name: str
    :param display_name: (optional) The column name that will be displayed in the csv/excel file. This is useful if you
    want to specify units to the user
    :type display_name: str
    """
    def __init__(self, key_name, display_name=None):
        self.key_name = key_name
        self.display_name = display_name


# If you want a morphometrics column to have a particular order or a display name with units (or both), you need to add
# it to this list. Otherwise, the column will be placed at the end of the morphometrics file
column_names_ordered = [
    Morphometrics_Column_Name('x0', 'x0 (px)'),
    Morphometrics_Column_Name('y0', 'y0 (px)'),
    Morphometrics_Column_Name('gratio'),
    Morphometrics_Column_Name('axon_area','axon_area (um^2)'),
    Morphometrics_Column_Name('axon_perimeter','axon_perimeter (um)'),
    Morphometrics_Column_Name('myelin_area', 'myelin_area (um^2)'),
    Morphometrics_Column_Name('axon_diam','axon_diam (um)'),
    Morphometrics_Column_Name('myelin_thickness','myelin_thickness (um)'),
    Morphometrics_Column_Name('axonmyelin_area','axonmyelin_area (um^2)'),
    Morphometrics_Column_Name('axonmyelin_perimeter','axonmyelin_perimeter (um)'),
]