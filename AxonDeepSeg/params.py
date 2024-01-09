import numpy as np

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
