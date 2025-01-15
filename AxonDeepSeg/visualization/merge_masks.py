
from pathlib import Path
import numpy as np

import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.ads_utils import convert_path

def merge_masks(path_axon, path_myelin):
    '''
    Merge axon and myelin masks into a single image.
    Also saves the merged mask in the same folder.

    Parameters
    ----------
    path_axon : str or pathlib.Path
        Path to the axon mask.
    path_myelin : str or pathlib.Path
        Path to the myelin mask.

    Returns
    -------
    both : ndarray
        The merged axon and myelin masks.
    '''

    # If string, convert to Path objects
    path_axon = convert_path(path_axon)
    path_myelin = convert_path(path_myelin)
    axon = ads.imread(path_axon)
    myelin = ads.imread(path_myelin)

    both = (axon/255)*255 + (myelin/255)*127
    
    # save the mask
    output_fname = path_axon.name.replace('axon', 'axonmyelin')
    path_folder = path_axon.parent
    ads.imwrite(path_folder / output_fname, both)

    return both
