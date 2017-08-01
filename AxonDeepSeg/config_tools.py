'''
Set the config variable.
'''

import ConfigParser as cp
import os
import json
import collections
import numpy as np
import time
import copy


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
    
    return {
        "network_n_classes": 3,
        "network_thresholds": [0, 0.2, 0.8],
        "network_batch_size": 8,
        "network_dropout": 0.75,
        "network_depth": 4,
        "network_convolution_per_layer": [2,2,2,2],
        "network_size_of_convolutions_per_layer": [[5,5,5],[3,3,3],[3,3,3],[3,3,3]],
        "network_features_per_convolution": [[[1,16],[16,16]], 
                                             [[16,32],[32,32]], 
                                             [[32,64],[64,64]],
                                             [[64,128],[128,128]]
                                            ],
        "network_trainingset": 'SEM_3classes_reduced',
        "network_downsampling": 'convolution',
        "network_data_augmentation": {'type':'all', 
                                      'transformations':{'0_shifting':True, '1_rescaling':True,
                                                         '2_random_rotation':False, '3_elastic':True, '4_flipping':True,
                                                         '5_noise_addition':False}
                                     },
        "network_learning_rate": 0.001,
        "network_batch_norm": True,
        "network_trainingset_patchsize": 256,
        "network_batch_norm_decay": 0.7,
        "network_additional_parameters":{'learning_rate_decay_activate':True,
                                         'learning_rate_decay_period':10,
                                         'learning_rate_decay_rate':0.99,
                                         'batch_norm_decay_decay_activate':True,
                                         'batch_norm_decay_ending_decay':0.9,
                                         'batch_norm_decay_decay_period':50
                                        },
        "network_weighted_cost": True,
        "network_weighted_cost_parameters":{'balanced_activate':True,
                                            'balanced_weights':[1.1, 1, 1.3],
                                            'boundaries_activate':False,
                                            'boundaries_sigma':2
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

############################################ For submission generator

def rec_update(elem, update_dict):
    if type(elem) == dict:
        return update_config(elem,update_dict)
    elif type(elem) == list:
        return [rec_update(e, update_dict) for e in elem]
    else:
        return None

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
            
            
def grid_config(L_struct, dict_params, base_config = default_configuration()):
    '''
    L_struct is a list of structures parameters in dictionnaries for the configuration file. It must contain at least the number of convolution per layer, the size of each kernel, and the nested list of number of features per layer.
    '''
    # First we create the different structures from the list
    base_config = update_config(default_configuration(), base_config) # We complete the base configuration if needed.
    
    L_configs = []

    for structure in L_struct:
        base_config.update(generate_struct(structure))
        L_configs.append(base_config)
        
                
    # Then we create the grid thanks to the params.
    for param, L_values in dict_params.iteritems():
        temp_config = L_configs
        L_configs = []
        if isinstance(L_values, collections.Iterable):            
            for v in L_values:
                rec_update(temp_config, {param:v})
                L_configs.append(copy.deepcopy(temp_config))
        # If it's just a value we just take this value
        else:
            rec_update(temp_config, {param:L_values})
            L_configs.append(copy.deepcopy(temp_config))

    # Finally we flatten the resulting nested list and we return a dictionnary with each key being the name of a model and the value being the configuration dictionnary
    L_configs = list(flatten(L_configs))
    
    #config_names = [generate_name_config(config)+'_'+str(i)+'-'+str(int(time.time()))[-3:] for i,config in enumerate(L_configs)]
    #print L_configs
    return {generate_name_config(config)+'_'+str(i)+'-'+str(int(time.time()))[-4:]:config for i,config in enumerate(L_configs)}
            

## ----------------------------------------------------------------------------------------------------------------

def generate_features(depth,network_first_num_features,features_augmentation,network_convolution_per_layer):

    increment = int(float(features_augmentation[1:]))

    if str(features_augmentation[0]) == 'p':
        # Add N features at each convolution layer.
        first_conv = [[1,network_first_num_features]]
        temp = [[network_first_num_features+i*increment,network_first_num_features+(i+1)*increment] 
                                for i in range(network_convolution_per_layer[0])[1:]]
        first_layer = first_conv + temp
        last_layer = first_layer
        network_features_per_convolution = [first_layer]

        for cur_depth in range(depth)[1:]:

            first_conv = [[last_layer[-1][-1],last_layer[-1][-1]+increment]]
            temp = [[last_layer[-1][-1]+i*increment,last_layer[-1][-1]+(i+1)*increment] for i in range(network_convolution_per_layer[cur_depth])[1:]]
            current_layer = first_conv+temp
            network_features_per_convolution = network_features_per_convolution + [current_layer]

            last_layer = current_layer

    elif str(features_augmentation[0]) == 'x':
        # Multiply the number of features by N at each "big layer".
        
        first_conv = [[1,network_first_num_features]]
        temp = [[network_first_num_features,network_first_num_features] 
                                for i in range(network_convolution_per_layer[0]-1)]
        first_layer = first_conv + temp
        last_layer = first_layer
        network_features_per_convolution = [first_layer]
        for cur_depth in range(depth)[1:]:
            first_conv = [[last_layer[-1][-1],last_layer[-1][-1]*increment]]
            temp = [[last_layer[-1][-1]*increment,last_layer[-1][-1]*increment] for i in range(network_convolution_per_layer[cur_depth]-1)]
            current_layer = first_conv+temp
            network_features_per_convolution = network_features_per_convolution + [current_layer]

            last_layer = current_layer

    else:
        raise 'Invalid input : please for features_augmentation' 
                                                 

    return network_features_per_convolution

## ----------------------------------------------------------------------------------------------------------------

def generate_name_config(config):
    
    name = ''
    
    # Downsampling
    if config['network_downsampling'] == 'convolution':
        name += 'cv_'
    elif config['network_downsampling'] == 'maxpooling':
        name += 'mp_'
            
    # Number of classes
    
    name += str(config['network_n_classes']) + 'c_'
    
    # Depth
    name += 'd' + str(config['network_depth']) + '_'

    # Number of convolutions per layer
    # Here we make the supposition that the number of convo per layer is the same for every layer
    name += 'c' + str(config['network_convolution_per_layer'][1]) + '_'

    # Size of convolutions per layer
    # Here we make the supposition that the size of convo is the same for every layer
    name += 'k' + str(config['network_size_of_convolutions_per_layer'][1][0]) + '_'

    # We don't mention the batch size anymore as we are doing 8 by default

    # Channels augmentation
    #name += str(L_struct['features_augmentation']) + '-'
    #name += str(L_struct['network_first_num_features'])  

    # We return a tuple
    return name

def generate_struct(dict_struct):
   
    network_feature_per_convolution = generate_features(depth=len(dict_struct['structure']),
                                                        network_first_num_features=dict_struct['network_first_num_features'],
                                                        features_augmentation=dict_struct['features_augmentation'],
                                                       network_convolution_per_layer=[len(e) for e in dict_struct['structure']]
                                                       )

    
    return {'network_depth':len(dict_struct['structure']),
            'network_features_per_convolution':network_feature_per_convolution,
            'network_size_of_convolutions_per_layer':dict_struct['structure'],
            'network_convolution_per_layer':[len(e) for e in dict_struct['structure']]
           }