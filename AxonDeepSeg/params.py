import numpy as np

intensity = {
    'binary': np.iinfo(np.uint8).max,
    'axon': np.iinfo(np.uint8).max,
    'myelin': np.iinfo(np.uint8).max // 2,
    'background': np.iinfo(np.uint8).min
}

# morphometrics column names
column_names = np.array(
    [],
    dtype=[
        ('x0 (px)', 'f4'),
        ('y0 (px)', 'f4'),
        ('gratio', 'f4'),
        ('axon_area (um\u00b2)', 'f4'), # unicode for ^2
        ('axon_perimeter (um)', 'f4'),
        ('myelin_area (um\u00b2)', 'f4'),
        ('axon_diam (um)', 'f4'),
        ('myelin_thickness (um)', 'f4'),
        ('axonmyelin_area (um\u00b2)', 'f4'),
        ('axonmyelin_perimeter (um)', 'f4'),
        ('solidity', 'f4'),
        ('eccentricity', 'f4'),
        ('orientation', 'f4'),
    ]
)
