import numpy as np
import os
import imageio
import AxonDeepSeg.ads_utils

def merge_masks(path_axon,path_myelin):

    axon = imageio.imread(path_axon)
    myelin = imageio.imread(path_myelin)
    
    both = (axon/255)*255 + (myelin/255)*127
    
    # get main path
    path_folder, file_name = os.path.split(path_axon)

    # save the masks
    imageio.imwrite(os.path.join(path_folder,'axon_myelin_mask.png'),both)

    return both
