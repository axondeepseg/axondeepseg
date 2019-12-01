import numpy as np

intensity = {
    'binary': np.iinfo(np.uint8).max,
    'axon': np.iinfo(np.uint8).max,
    'myelin': np.iinfo(np.uint8).max // 2,
    'background': np.iinfo(np.uint8).min
}
