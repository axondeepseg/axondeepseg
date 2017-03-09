# -*- coding: utf-8 -*-

# guide


import json
import os


# Description du fichier config :
# network_learning_rate : float : No idea, but certainly linked to the back propagation ? Default : 0.0005.
# network_n_classes : int : number of labels in the output. Default : 2.
# dropout : float : between 0 and 1 : percentage of neurons we want to keep. Default : 0.75.
# network_depth : int : number of layers WARNING : factualy, there will be 2*network_depth layers. Default : 6.
# network_convolution_per_layer : list of int, length = network_depth : number of convolution per layer. Default : [1 for i in range(network_depth)].
# network_size_of_convolutions_per_layer : list of lists of int [number of layers[number_of_convolutions_per_layer]] : Describe the size of each convolution filter.
# Default : [[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)].

# network_features_per_convolution : list of lists of int [number of layers[number_of_convolutions_per_layer[2]] : Numer of different filters that are going to be used.
# Default : [[[64,64] for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]. WARNING ! network_features_per_convolution[k][1] = network_features_per_convolution[k+1][0].

filename = '/Users/piant/axondeepseg/AxonDeepSeg/init_config_network.json'

network_learning_rate = 0.0005
network_n_classes = 2
dropout = 0.75
network_depth = 3
network_convolution_per_layer = [1 for i in range(network_depth)]
network_size_of_convolutions_per_layer = [[3 for k in range(network_convolution_per_layer[i])] for i in
                                          range(network_depth)]
network_features_per_convolution = [[[64,64] for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]

"""
network_learning_rate = 0.0005
network_n_classes = 2
dropout = 0.75
network_depth = 6
network_convolution_per_layer = [2 for i in range(network_depth)]
network_size_of_convolutions_per_layer = [[3 for k in range(network_convolution_per_layer[i])] for i in
                                          range(network_depth)]
network_features_per_convolution = [[[1,64],[64,64]],[[64,128],[128,128]],[[128,256],[256,256]],[[256,512],[512,512]],[[512,1024],[1024,1024]],[[1024,2048],[2048,2048]]]
"""

config = {
    'network_learning_rate': 0.0005,
    'network_n_classes': 2,
    'dropout': 0.75,
    'network_depth': 3,
    'network_convolution_per_layer': [1 for i in range(network_depth)],
    'network_size_of_convolutions_per_layer': [[3 for k in range(network_convolution_per_layer[i])] for i in
                                               range(network_depth)],
    'network_features_per_convolution': [[[64,64] for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]
}
"""
config = {
    'network_learning_rate': 0.0005,
    'network_n_classes': 2,
    'dropout': 0.75,
    'network_depth': 6,
    'network_convolution_per_layer': [2 for i in range(network_depth)],
    'network_size_of_convolutions_per_layer': [[3 for k in range(network_convolution_per_layer[i])] for i in
                                               range(network_depth)],
    'network_features_per_convolution': [[[1,64],[64,64]],[[64,128],[128,128]],[[128,256],[256,256]],[[256,512],[512,512]],
                                    [[512,1024],[1024,1024]],[[1024,2048],[2048,2048]]]
}
"""
# Edit and read the config

with open(filename, 'w+') as f:
    json.dump(config, f, indent=2)

with open(filename, 'r') as fd:
	config_network = json.loads(fd.read())

learning_rate = config_network.get("network_learning_rate", 0.0005)
n_classes = config_network.get("network_n_classes", 2)
dropout = config_network.get("network_dropout", 0.75)
depth = config_network.get("network_depth", 6)
number_of_convolutions_per_layer = config_network.get("network_convolution_per_layer", [1 for i in range(depth)])
size_of_convolutions_per_layer =  config_network.get("network_size_of_convolution_per_layer",[[3 for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])
features_per_convolution = config_network.get("network_features_per_convolution",[[[64,64] for k in range(number_of_convolutions_per_layer[i])] for i in range(depth)])

# print(learning_rate,n_classes,dropout,depth,number_of_convolutions_per_layer,size_of_convolutions_per_layer,features_per_convolution)

# training
path_training = '/Users/piant/axondeepseg/trainingset'
path_model = '/Users/piant/axondeepseg/models/Unet_changed'
#
from AxonDeepSeg.new_script_network import train_model
train_model(path_training, path_model, config_network)

from learn_model import train_model
#train_model(path_training, path_model, learning_rate=0.0005)