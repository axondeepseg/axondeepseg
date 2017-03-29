# -*- coding: utf-8 -*-

# guide


from sklearn.metrics import recall_score

network_learning_rate = 0.0005
network_n_classes = 2
dropout = 0.75
network_depth = 4
network_convolution_per_layer = [1 for i in range(network_depth)]
network_size_of_convolutions_per_layer = [[5],[3],[5],[9]]#[[3 for k in range(network_convolution_per_layer[i])] for i in range(network_depth)]
network_features_per_convolution = [[[1,64]],[[64,128]],[[128,256]],[[256,512]]]#,[[256,512],[512,512]],[[512,1024],[1024,1024]],[[1024,2048],[2048,2048]]]

config = {
    'network_learning_rate': 0.0005,
    'network_n_classes': 2,
    'network_dropout': 0.75,
    'network_depth': 3,
    'network_convolution_per_layer': [1 for i in range(network_depth)],
    'network_size_of_convolutions_per_layer': network_size_of_convolutions_per_layer,
    'network_features_per_convolution': network_features_per_convolution
}


# training
path_training = './../trainingset'

from network_optimization import Trainer


network_model = {'model_name': 'Unet',
                 'model_hyperparam':{'network_learning_rate': 0.0005,
                                     'network_n_classes': 2,
                                     'network_dropout': 0.75,
                                     'network_depth': 3,
                                     'network_convolution_per_layer': [1 for i in range(network_depth)],
                                     'network_size_of_convolutions_per_layer': network_size_of_convolutions_per_layer,
                                     'network_features_per_convolution': network_features_per_convolution}}


param_training = {'number_of_epochs': 500, 'patch_size': [256,256],
                  'results_path' : './../trainingset/results',
                  'hyperopt': {'algo':tpe.suggest,        # Grid Search algorithm
                                'nb_eval':1,               # Nb max of param test
                                'fct': recall_score,       # Objective function
                                'eval_factor': 1,           # Evaluation rate
                                'ratio_dataset_eval':0.15,   # Ratio of training dataset dedicated to hyperParam validation
                                'ratio_img_eval':1.0,       # Ratio of patch per validation image
                                'ratio_img_train':0.3}}      # Ratio of patch per training image

my_trainer = Trainer(path_trainingset=path_training,
                     classifier_model=network_model,
                     param_training=param_training)

my_trainer.hyperparam_optimization()
my_trainer.set_hyperopt_train(path_trainingset=path_training)
