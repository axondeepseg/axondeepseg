# coding: utf-8

import json
import os
import shutil
import tensorflow as tf
from shutil import copy
import pytest

from AxonDeepSeg.train_network import train_model

class TestCore(object):
    def setup(self):
        # reset the tensorflow graph for new training
        tf.reset_default_graph()

        self.fullPath = os.path.dirname(os.path.abspath(__file__))

        self.modelPath = os.path.join(self.fullPath, '__test_files__/__test_training_files__/Model')
        self.configPath = os.path.join(self.fullPath, '__test_files__/__test_training_files__/Model/config_network.json')
        self.trainingPath = os.path.join(self.fullPath, '__test_files__/__test_training_files__')

        if not os.path.exists(self.modelPath):
            os.makedirs(self.modelPath)

        self.config = {
    
            # General parameters:    
            "n_classes": 3,
            "thresholds": [0, 0.2, 0.8],    
            "trainingset_patchsize": 256,    
            "trainingset": "SEM_3c_256",    
            "batch_size": 2,
            "save_epoch_freq": 1,

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
            "learning_rate_decay_period": 4, 
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

        if os.path.exists(self.configPath):
            with open(self.configPath, 'r') as fd:
                self.config_network = json.loads(fd.read())
        else: # There is no config file for the moment
            with open(self.configPath, 'w') as f:
                json.dump(self.config, f, indent=2)
            with open(self.configPath, 'r') as fd:
                self.config_network = json.loads(fd.read())

    @classmethod
    def teardown_class(cls):
        fullPath = os.path.dirname(os.path.abspath(__file__))

        modelPath = os.path.join(fullPath, '__test_files__/__test_training_files__/Model')

        if os.path.exists(modelPath) and os.path.isdir(modelPath):
            shutil.rmtree(modelPath)

    #--------------train_model tests--------------#
    @pytest.mark.integration
    def test_train_model_runs_succesfully_for_simplified_case(self):
    # Note: This test is simply a mock test to ensure that the pipeline runs succesfully, and is not
    # a test of the quality of the model itself.

        train_model(self.trainingPath, self.modelPath, self.config_network, debug_mode=True)

        expectedFiles = ["checkpoint",
                         "config_network.json",
                         "evolution_stats.pkl",
                         "evolution.pkl",
                         "model.ckpt.data-00000-of-00001",
                         "model.ckpt.index",
                         "model.ckpt.meta",
                         "report.txt"
                        ]
        existingFiles = os.listdir(self.modelPath)

        for fileName in expectedFiles:
            assert fileName in existingFiles
