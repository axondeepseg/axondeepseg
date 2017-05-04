# Network training with cleaned branch :

# Explains how to train, visualize and segment on a new image

# download + unzip example data
# https://www.dropbox.com/s/juybkrzzafgxkuu/victor.zip?dl=1

###########################
# Load a .json config file containing :

    # network_learning_rate : float : No idea, but certainly linked to the back propagation ? Default : 0.0005.
    
    # network_n_classes : int : number of labels in the output. Default : 2.
    
    # network_dropout : float : between 0 and 1 : percentage of neurons we want to keep. Default : 0.75.
    
    # network_depth : int : number of layers WARNING : factualy, there will be 2*network_depth layers. Default : 6.
    
    # network_convolution_per_layer : list of int, length = network_depth : number of convolution per layer. Default : [1 for i in range(network_depth)].
    
    # network_size_of_convolutions_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer]] : Describe the size of each convolution filter. 
    # Default : [[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)].
    
    # network_features_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer[2]] : Numer of different filters that are going to be used.
    # Default : [[64 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]. WARNING ! network_features_per_layer[k][1] = network_features_per_layer[k+1][0].
    
    # network_trainingset : string : describe the trainingset for the network.

    # network_downsampling : string 'maxpooling' or 'convolution' : the downsampling method. 

    # network_thresholds : list of float in [0,1] : the thresholds for the ground truthes labels.

    # network_weighted_cost : boolean : whether we use weighted cost for training or not.
###########################

# Example

# 1/ Build the dataset for training.
# input data to build the training set
path_data = 'mypath_raw_data'

# output path of training data path
path_training = 'mypath_trainingset'

from AxonDeepSeg.data.dataset_building import build_dataset
build_dataset(path_data, path_training, trainRatio=0.80)


# 2/ Load the config file and train the network
import json
import os

# output path for trained U-Net
path_model = 'mypath_models/Unet_new'

# optional input path of a model to initialize the training
path_model_init = 'mypath_models/Unet_init'

path_configfile = 'mypath_configfile'

if not os.path.exists(path_model):
    os.makedirs(path_model)

with open(path_configfile, 'r') as fd:
    config_network = json.loads(fd.read())

# OPTIONAL : specify the gpu one wants to use.
gpu_device = 'gpu:0' # or gpu_device = 'gpu:1' these are the only two possible targets for now.

from AxonDeepSeg.train_network import train_model
train_model(path_training, path_model, config_network, path_model_init=None, gpu = gpu_device)


# 3/ Local Visualization
from AxonDeepSeg.evaluation.visualization import visualize_training
# sub-option 1 : if you do not have an initial model
visualize_training(path_model)
# sub-option 2 : if you have an initial model
visualize_training(path_model, path_model_init)


# 4/ Apply the model to segment one image
path_target_image = 'mypath_target_image'

from AxonDeepSeg.apply_model import axon_segmentation
axon_segmentation(path_my_data, path_model, config_network)


# 5/ Myelin segmentation from axon segmentation
from AxonDeepSeg.apply_model import myelin
# Axon_segmentation() has to be already runned
myelin(path_my_data)


# 6/ Visualization of the results
from AxonDeepSeg.evaluation.visualize import visualize_segmentation
visualize_segmentation(path_my_data)


# 7/ Print the features. 

