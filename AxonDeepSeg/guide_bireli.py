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

# network_trainingset : string : String describing the dataset used for the training.


filename = '/robert_config_network.json'

network_learning_rate = 0.0005
network_n_classes = 2
dropout = 0.75
network_depth = 6
network_convolution_per_layer = [3 for i in range(network_depth)]
network_size_of_convolutions_per_layer = [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]]#[[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]
network_features_per_convolution = [[[1,10],[10,20],[20,30]],[[30,40],[40,50],[50,60]],[[60,70],[70,80],[80,90]],[[90,100],[100,110],[110,120]],
                                    [[120,130],[130,140],[140,150]],[[150,160],[160,170],[170,180]]]
trainingset = 'CARS_tot'

downsampling = 'convolution'

thresholds = [0,0.5]

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


# Edit and read the config
"""if not os.path.exists(repname):
    os.makedirs(repname)

with open(repname+filename, 'w+') as f:
    json.dump(config, f, indent=2)

with open(repname+filename, 'r') as fd:
    config_network = json.loads(fd.read())"""

# training
path_training = '/home/piant_local/piant/python_rep/axondeepseg/trainingset/'+trainingset
path_model = '/home/piant_local/piant/python_rep/axondeepseg/models/2classes'
path_model_init = '/home/piant_local/piant/python_rep/axondeepseg/models/2classes'

if not os.path.exists(path_model):
    os.makedirs(path_model)

   
with open(path_model+filename, 'w') as f:
    json.dump(config, f, indent=2)

with open(path_model+filename, 'r') as fd:
    config_network = json.loads(fd.read())

from final_network import train_model
train_model(path_training, path_model, config_network,path_model_init=None,gpu='gpu:0')

#from new_script_network import train_model
#train_model(path_training, path_model, config_network,path_model_init=None)

#from AxonDeepSeg.learn_model import train_model
#train_model(path_training, path_model, learning_rate=0.0005)
