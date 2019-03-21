
from pathlib import Path
import numpy as np
import imageio

import AxonDeepSeg.ads_utils

def merge_masks(path_axon, path_myelin):

    axon = imageio.imread(path_axon)
    myelin = imageio.imread(path_myelin)

    both = (axon/255)*255 + (myelin/255)*127

    # get main path
    path_folder = path_axon.parent

    # save the masks
    imageio.imwrite(path_folder / 'axon_myelin_mask.png', both)

    return both
