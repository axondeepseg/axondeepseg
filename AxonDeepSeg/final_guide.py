# -*- coding: utf-8 -*-

# guide

import json
import os

########## HEADER ##########
# Config file description :

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

filename = '/config_network.json'

network_learning_rate = 0.0005
network_n_classes = 2
dropout = 0.75
network_depth = 6
network_convolution_per_layer = [3 for i in range(network_depth)]
network_size_of_convolutions_per_layer = [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5,
                                                                                                  5], ]  # [[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]
network_features_per_convolution = [[[1, 10], [10, 20], [20, 30]], [[30, 40], [40, 50], [50, 60]],
                                    [[60, 70], [70, 80], [80, 90]], [[90, 100], [100, 110], [110, 120]],
                                    [[120, 130], [130, 140], [140, 150]], [[150, 160], [160, 170], [170, 180]]]
trainingset = 'CARS_tot'

downsampling = 'convolution'

thresholds = [0, 0.5]

weighted_cost = True

config = {
    'network_learning_rate': network_learning_rate,
    'network_n_classes': network_n_classes,
    'network_dropout': dropout,
    'network_depth': network_depth,
    'network_convolution_per_layer': network_convolution_per_layer,
    'network_size_of_convolutions_per_layer': network_size_of_convolutions_per_layer,
    'network_features_per_convolution': network_features_per_convolution,
    'network_trainingset': trainingset,
    'network_downsampling': downsampling,
    'network_thresholds': thresholds,
    'network_weighted_cost': weighted_cost
}

# training paths
path_training = '/Users/piant/axondeepseg_data/trainingset/' + trainingset
path_model = '/Users/piant/axondeepseg_data/models/TEST'
path_model_init = '/Users/piant/axondeepseg_data/models/TEST'

# Create the folder for the model and (read/edit/save) the config file in it.
if not os.path.exists(path_model):
    os.makedirs(path_model)

with open(path_model + filename, 'w') as f:
    json.dump(config, f, indent=2)

with open(path_model + filename, 'r') as fd:
    config_network = json.loads(fd.read())

# Training
from train_network import train_model

train_model(path_training, path_model, config_network, path_model_init=None)
