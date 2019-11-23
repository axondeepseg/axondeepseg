'''
Set the config variable.
'''

import configparser as cp
from pathlib import Path
import json
import collections
import time
import copy
import AxonDeepSeg.ads_utils


config = cp.RawConfigParser()
file_path = Path(__file__).resolve().parent
config.read(file_path / 'data_management' / 'config.cfg')


#### Network config file management. ####

def validate_config(config):

    """ Check the config file keys
    :param config: Dictionary containing the parameters of the network.
    :return: Boolean. True if the configuration is valid compared to the default configuration, else False.
    """

    keys = list(default_configuration().keys())
    for key in list(config.keys()):
        if not key in keys:
            return False
    return True


def default_configuration():
    """
    Generate the default configuration for the training parameters.
    :return: Dictionary, the default configuration parameters.
    """

    tmp = {'batch_norm_decay_decay_activate': True,
     'batch_norm_decay_decay_period': 24000,
     'batch_norm_decay_starting_decay': 0.7,
     'batch_norm_decay_ending_decay': 0.9,
     'learning_rate_decay_activate': True,
     'learning_rate_decay_period': 16000,
     'learning_rate_decay_rate': 0.99, # Only used for exponential decay
     'learning_rate_decay_type': 'polynomial',
     'batch_norm_activate': True,
     'batch_size': 8,
     'convolution_per_layer': [3, 3, 3, 3],
     'da-3-elastic-activate': True,
     'da-elastic-alpha_max': 9,
     'da-elastic-order': 3,
     'da-4-flipping-activate': True,
     'da-flipping-order': 4,
     'da-gaussian_blur-activate': True,
     'da-gaussian_blur-order': 6,
     'da-gaussian_blur-sigma_max': 1.5,
     'da-5-noise_addition-activate': False,
     'da-noise_addition-order': 5,
     'da-2-random_rotation-activate': False,
     'da-random_rotation-high_bound': 89,
     'da-random_rotation-low_bound': 5,
     'da-random_rotation-order': 2,
     'da-1-rescaling-activate': False,
     'da-rescaling-factor_max': 1.2,
     'da-rescaling-order': 1,
     'da-0-shifting-activate': True,
     'da-shifting-order': 0,
     'da-shifting-percentage_max': 0.1,
     'da-type': 'all',
     'depth': 4,
     'downsampling': 'convolution',
     'dropout': 0.75,
     'features_per_convolution': [[[1, 16], [16, 16], [16, 16]],
      [[16, 32], [32, 32], [32, 32]],
      [[32, 64], [64, 64], [64, 64]],
      [[64, 128], [128, 128], [128, 128]]],
     'learning_rate': 0.001,
     'n_classes': 3,
     'size_of_convolutions_per_layer': [[5, 5, 5],
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3]],
     'weighted_cost-activate': True,
     'weighted_cost-balanced_activate': True,
     'weighted_cost-balanced_weights': [1.1, 1, 1.3],
     'weighted_cost-boundaries_activate': False,
     'weighted_cost-boundaries_sigma': 2,
   'thresholds': [0, 0.2, 0.8],
   'trainingset': 'SEM_3c_512',
   'trainingset_patchsize': 512,
   'balanced_weights': [1.1, 1, 1.3],
   'dataset_mean': 120.95, # Not used right now for preprocessing, we do it on a per image basis.
   'dataset_variance': 60.23 # Not used right now for preprocessing, we do it on a per image basis. THIS SHOULD BE STD
           }

    return tmp


def update_config(d, u):
    for k, v in list(u.items()):
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
        tmp = copy.deepcopy(base_config)
        tmp.update(generate_struct(structure))
        L_configs.append(tmp)
                
                
    # Then we create the grid thanks to the params.
    for param, L_values in list(dict_params.items()):
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
        raise ValueError('Invalid features_augmentation value. Must begin with x or p, and be followed by an integer.' )
                                                 

    return network_features_per_convolution

## ----------------------------------------------------------------------------------------------------------------

def generate_name_config(config):
    
    name = ''
    
    # Downsampling
    if config['downsampling'] == 'convolution':
        name += 'cv_'
    elif config['downsampling'] == 'maxpooling':
        name += 'mp_'
            
    # Number of classes
    
    name += str(config['n_classes']) + 'c_'
    
    # Depth
    name += 'd' + str(config['depth']) + '_'

    # Number of convolutions per layer
    # Here we make the supposition that the number of convo per layer is the same for every layer
    name += 'c' + str(config['convolution_per_layer'][1]) + '_'

    # Size of convolutions per layer
    # Here we make the supposition that the size of convo is the same for every layer
    name += 'k' + str(config['size_of_convolutions_per_layer'][1][0]) + '_'

    # We don't mention the batch size anymore as we are doing 8 by default

    # Channels augmentation
    #name += str(L_struct['features_augmentation']) + '-'
    #name += str(L_struct['network_first_num_features'])  

    # We return a tuple
    return name

def generate_struct(dict_struct):
   
    network_feature_per_convolution = generate_features(depth=len(dict_struct['structure']),
                                                        network_first_num_features=dict_struct['first_num_features'],
                                                        features_augmentation=dict_struct['features_augmentation'],
                                                       network_convolution_per_layer=[len(e) for e in dict_struct['structure']]
                                                       )

    
    return {'depth':len(dict_struct['structure']),
            'features_per_convolution':network_feature_per_convolution,
            'size_of_convolutions_per_layer':dict_struct['structure'],
            'convolution_per_layer':[len(e) for e in dict_struct['structure']]
           }
