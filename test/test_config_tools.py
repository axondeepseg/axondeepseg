# coding: utf-8

import pytest
import json
import os
import shutil
from AxonDeepSeg.config_tools import *

class TestCore(object):
    def setup(self):

        self.config = {
    
            # General parameters:    
            "n_classes": 3,
            "thresholds": [0, 0.2, 0.8],    
            "trainingset_patchsize": 256,    
            "trainingset": "SEM_3c_256",    
            "batch_size": 8,

            # Network architecture parameters:     
            "depth": 2,
            "convolution_per_layer": [2, 2],
            "size_of_convolutions_per_layer": [[3, 3], [3, 3]],
            "features_per_convolution": [[[1, 5], [5, 5]], [[5, 10], [10, 10]]],
            "downsampling": "maxpooling",
            "dropout": 0.75, 
                
            # Learning rate parameters:    
            "learning_rate": 0.001,    
            "learning_rate_decay_activate": True,    
            "learning_rate_decay_period": 24000, 
            "learning_rate_decay_type": "polynomial", 
            "learning_rate_decay_rate": 0.99,
                
            # Batch normalization parameters:     
            "batch_norm_activate": True,     
            "batch_norm_decay_decay_activate": True,    
            "batch_norm_decay_starting_decay": 0.7, 
            "batch_norm_decay_ending_decay": 0.9, 
            "batch_norm_decay_decay_period": 16000,
                    
            # Weighted cost parameters:    
            "weighted_cost-activate": True, 
            "weighted_cost-balanced_activate": True, 
            "weighted_cost-balanced_weights": [1.1, 1, 1.3], 
            "weighted_cost-boundaries_sigma": 2, 
            "weighted_cost-boundaries_activate": False, 
                
            # Data augmentation parameters:    
            "da-type": "all", 
            "da-2-random_rotation-activate": False, 
            "da-5-noise_addition-activate": False, 
            "da-3-elastic-activate": True, 
            "da-0-shifting-activate": True, 
            "da-4-flipping-activate": True, 
            "da-1-rescaling-activate": False    
            }

        # Create temp folder
        self.fullPath = os.path.dirname(os.path.abspath(__file__))
        self.tmpPath = os.path.join(self.fullPath, '__tmp__')
        if not os.path.exists(self.tmpPath):
            os.makedirs(self.tmpPath)

    @classmethod
    def teardown_class(cls):
        fullPath = os.path.dirname(os.path.abspath(__file__))
        tmpPath = os.path.join(fullPath, '__tmp__')
        if os.path.exists(tmpPath):
            shutil.rmtree(tmpPath)
        pass

    #--------------config_tools.py tests--------------#
    @pytest.mark.unittest
    def test_validate_config_for_demo_config(self):
        assert validate_config(self.config)

    def test_validate_config_for_invalid_config(self):
        
        invalidConfig = {
            "1nval1d_k3y": 0
        }

        assert not validate_config(invalidConfig)

    @pytest.mark.unittest
    def test_generate_config_creates_valid_config(self):
        generatedConfig = generate_config()

        assert validate_config(generatedConfig)

    @pytest.mark.unittest
    def test_generate_config_with_config_path(self):
        # Create temp config file
        fullPath = os.path.dirname(os.path.abspath(__file__))
        configPath = os.path.join(fullPath, '__temp_files__/config_network.json')

        if os.path.exists(configPath):
            os.remove(configPath)
            with open(configPath, 'w') as f:
                json.dump(self.config, f, indent=2)
        else: # There is no config file for the moment
            with open(configPath, 'w') as f:
                json.dump(self.config, f, indent=2)
        

        generatedConfig = generate_config(config_path = configPath)
        assert generatedConfig != generate_config()
        assert validate_config(generatedConfig)

        if os.path.exists(configPath):
            os.remove(configPath)

    @pytest.mark.unittest
    def test_generate_config_with_config_path_and_invalid_onfig(self):
        invalidConfig = {
            "1nval1d_k3y": 0
        }
        
        configPath = os.path.join(self.tmpPath, 'config_network.json')

        if os.path.exists(configPath):
            os.remove(configPath)
            with open(configPath, 'w') as f:
                json.dump(invalidConfig, f, indent=2)
        else: # There is no config file for the moment
            with open(configPath, 'w') as f:
                json.dump(invalidConfig, f, indent=2)
 
        with pytest.raises(ValueError):
            generatedConfig = generate_config(config_path = configPath)

        if os.path.exists(configPath):
            os.remove(configPath)

    @pytest.mark.unittest
    def test_grid_config_feature_augmentation_x(self):
        # Sample L_struct and dict_params values below taken from guide_bireli.ipynb
        # 'features_augmentation':'x2'
        L_struct = [{'structure':[[5,5,5], [3,3,3], [3,3,3], [3,3,3]], 'features_augmentation':'x2', 'first_num_features':16}]
        dict_params = {'trainingset': ['SEM_3c_512'], 'trainingset_patchsize': 512, 'learning_rate_decay_period':24000 }
        
        config_list = grid_config(L_struct, dict_params, base_config = self.config)
        
        for key in config_list.keys():
            assert validate_config(config_list[key])

    @pytest.mark.unittest
    def test_grid_config_feature_augmentation_p(self):
        # Sample L_struct and dict_params values below taken from guide_bireli.ipynb
        # 'features_augmentation':'p2'
        L_struct = [{'structure':[[5,5,5], [3,3,3], [3,3,3], [3,3,3]], 'features_augmentation':'p2', 'first_num_features':16}]
        dict_params = {'trainingset': ['SEM_3c_512'], 'trainingset_patchsize': 512, 'learning_rate_decay_period':24000 }
        
        config_list = grid_config(L_struct, dict_params, base_config = self.config)
        
        for key in config_list.keys():
            assert validate_config(config_list[key])

    @pytest.mark.unittest
    def test_grid_config_feature_augmentation_invalid(self):
        # Sample L_struct and dict_params values below taken from guide_bireli.ipynb
        # 'features_augmentation':'d2'-> d is not a valid augmentation flag.
        L_struct = [{'structure':[[5,5,5], [3,3,3], [3,3,3], [3,3,3]], 'features_augmentation':'d2', 'first_num_features':16}]
        dict_params = {'trainingset': ['SEM_3c_512'], 'trainingset_patchsize': 512, 'learning_rate_decay_period':24000 }
        
        with pytest.raises(ValueError):
            config_list = grid_config(L_struct, dict_params, base_config = self.config)

    @pytest.mark.unittest
    def test_generate_name_config_convolution_downsampling_first_letters(self):
        tmpConfig = self.config
        tmpConfig['downsampling'] = 'convolution'

        configName = generate_name_config(tmpConfig)

        assert configName[:3] == 'cv_'
