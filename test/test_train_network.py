# coding: utf-8

import json
from pathlib import Path
import shutil

import keras.backend.tensorflow_backend as K


import pytest

from AxonDeepSeg.train_network import train_model


class TestCore(object):
    def setup(self):
        # reset the tensorflow graph for new training

        K.clear_session()
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent

        self.modelPath = (
            self.fullPath /
            '__test_files__' /
            '__test_training_files__' /
            'Model'
            )

        self.configPath = (
            self.fullPath /
            '__test_files__' /
            '__test_training_files__' /
            'Model' /
            'config_network.json'
            )

        self.trainingPath = (
            self.fullPath /
            '__test_files__' /
            '__test_training_files__'
            )

        self.modelPath.mkdir(parents=True, exist_ok=True)

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
            "features_per_convolution": [
                [[1, 5], [5, 5]],
                [[5, 10], [10, 10]]
                ],
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

        if not self.configPath.exists():
            with open(self.configPath, 'w') as f:
                json.dump(self.config, f, indent=2)

        with open(self.configPath, 'r') as fd:
            self.config_network = json.loads(fd.read())

    @classmethod
    def teardown_class(cls):
        fullPath = Path(__file__).resolve().parent

        modelPath = (
            fullPath /
            '__test_files__' /
            '__test_training_files__' /
            'Model'
            )

        if modelPath.is_dir():
            try:
                shutil.rmtree(modelPath)
            except OSError:
                print("Could not clean up {} - you may want to remove it manually.".format(modelPath))

    # --------------train_model tests-------------- #
    @pytest.mark.integration
    def test_train_model_runs_successfully_for_simplified_case(self):
        # Note: This test is simply a mock test to ensure that the pipeline
        # runs successfully, and is not a test of the quality of the model
        # itself.

        train_model(
            str(self.trainingPath),
            str(self.modelPath),
            self.config_network,
            debug_mode=True
            )

        expectedFiles = [
            "checkpoint",
            "config_network.json",
            "model.ckpt.data-00000-of-00001",
            "model.ckpt.index",
            "model.ckpt.meta"
            ]

        existingFiles = [f.name for f in self.modelPath.iterdir()]

        for fileName in expectedFiles:
            assert fileName in existingFiles
