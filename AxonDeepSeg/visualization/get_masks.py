import numpy as np
import pandas as pd
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio

def get_masks(path_prediction):

    prediction = imageio.imread(path_prediction)

    # compute the axon mask
    axon_prediction = prediction > 200

    # compute the myelin mask
    myelin_prediction = prediction > 100
    myelin_prediction = myelin_prediction ^ axon_prediction
    
    # get main path
    path_folder, file_name = os.path.split(path_prediction)

    # save the masks
    imageio.imwrite(os.path.join(path_folder,'axon_mask.png'),axon_prediction.astype(int))
    imageio.imwrite(os.path.join(path_folder,'myelin_mask.png'),myelin_prediction.astype(int))

    return axon_prediction, myelin_prediction

























