'''
Set the config variable.
'''

import ConfigParser as cp
import os
import json
import collections


config = cp.RawConfigParser()
config.read(os.path.dirname(__file__) + '/data/config.cfg')

# Global variables
path_axonseg = config.get("paths", "path_axonseg")
path_matlab = config.get("paths", "path_matlab")
general_pixel_size = float(config.get("variables", "general_pixel_size"))


#### Network config file management. ####

def validate_config(config):
    """ Check the config file keys """
    keys = config.keys()
    for key in default_configuration().keys():
        if not key in keys:
            return False
    return True


def default_configuration():
    """ Generate the default configuration."""
    depth = 4
    number_of_convolutions_per_layer = [1 for i in range(depth)]
    
    return {
        "network_n_classes": 2,
        "network_thresholds": [0, 0.5],
        "network_learning_rate": 0.0005,
        "network_batch_size": 8,
        "network_dropout": 0.75,
        "network_batch_norm_decay": 0.999,
        "network_depth": 4,
        "network_convolution_per_layer": number_of_convolutions_per_layer,
        "network_size_of_convolutions_per_layer": [[3 for k in range(number_of_convolutions_per_layer[i])] for i in
                                                   range(depth)],
        "network_features_per_convolution": [[[64, 64] for k in range(number_of_convolutions_per_layer[i])] for i in
                                             range(depth)],
        "network_trainingset": 'SEM_2classes_reduced',
        "network_weighted_cost": False,
        "network_downsampling": 'maxpooling',
        "network_batch_norm": True,
        "network_data_augmentation": {'type':'all', 
                                      'transformations':{'shifting':True, 'rescaling':True,
                                                         'random_rotation':True, 'elastic':True, 'flipping':True,
                                                         'noise_addition':True}
                                     }
    }


def update_config(d, u):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update_config(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
    


def generate_config(config_path=None):
    """ Generate the config file
    Input : 
        config_path : string : path to a json config file. If None, using the default configuration.
    Output :
        config : dict : the network config file.
    """
    config = default_configuration()
    if config_path != None:
        with open(config_path) as conf_file:
            user_conf = json.load(conf_file)
            config.update(user_conf)
    if not validate_config(config):
        raise ValueError('Invalid configuration file')

    return config

############################################
