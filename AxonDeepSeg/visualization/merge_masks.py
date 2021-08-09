
from pathlib import Path
import numpy as np

import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.ads_utils import convert_path

def merge_masks(path_axon, path_myelin, output_filename):
    print(path_axon)
    print(path_myelin)
    print(output_filename)
    # If string, convert to Path objects
    path_axon = convert_path(path_axon)

    axon = ads.imread(path_axon)
    myelin = ads.imread(path_myelin)

    both = (axon/255)*255 + (myelin/255)*127

    # get main path
    path_folder = path_axon.parent


    print(path_folder / output_filename)
    
    # save the masks
    ads.imwrite(path_folder / output_filename, both)

    return both
