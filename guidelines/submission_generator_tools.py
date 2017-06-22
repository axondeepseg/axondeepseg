import json
import os

def default_config():
    """Generate the default config dict"""
    network_learning_rate = 0.0005
    network_n_classes = 3
    dropout = 0.75
    network_depth = 4
    network_convolution_per_layer = [3 for i in range(network_depth)]
    network_size_of_convolutions_per_layer = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]  
    network_features_per_convolution =  [[[1, 10], [10, 20], [20, 30]], [[30, 40], [40, 50], [50, 60]],
                                        [[60, 70], [70, 80], [80, 90]], [[90, 100], [100, 110], [110, 120]]]
    trainingset = 'SEM_2classes_reduced'
    downsampling = 'maxpooling'
    thresholds = [0, 0.5]
    weighted_cost = False
    batch_size = 8

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
        'network_weighted_cost': weighted_cost,
        'network_batch_size': batch_size
    }
    return config

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

def generate_dict(  network_learning_rate = 0.0005,
                    network_n_classes = 2,
                    dropout = 0.75,
                    structure = [[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
                    features_augmentation = 'p10',
                    network_first_num_features = 10,
                    trainingset = 'SEM_2classes_reduced',
                    downsampling = 'maxpooling',
                    thresholds = [0, 0.5],
                    weighted_cost = False,
                    batch_size = 8):

    """ features_augmentation : string : first caracter p for addition or x for multiplication. Rest of the string : imcrementation value. """

    network_learning_rate = network_learning_rate
    network_n_classes = network_n_classes
    dropout = dropout
    
    trainingset = trainingset
    downsampling = downsampling
    thresholds = thresholds
    weighted_cost = weighted_cost

    network_first_num_features = network_first_num_features
    network_depth = len(structure)
    network_convolution_per_layer = [len(e) for e in structure]
    network_size_of_convolutions_per_layer = structure

    network_features_per_convolution = generate_features(network_depth,network_first_num_features,features_augmentation,network_convolution_per_layer)

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
        'network_weighted_cost': weighted_cost,
        'network_batch_size': batch_size
    }
    
    # name generation
    
    name = ''
        
    # Downsampling
    if config['network_downsampling'] == 'convolution':
        name += 'cv_'
    elif config['network_downsampling'] == 'maxpooling':
        name += 'mp_'

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
    name += str(features_augmentation) + '-'
    name += str(network_first_num_features)

    # We return a tuple
    return {name:config}

## ----------------------------------------------------------------------------------------------------------------

def generate_heliosjob(bashname,configfile,config,path_trainingset,path_model,path_model_init = None):
    """Generate config file given a config dict. Generate the corresponding submission."""

    if not os.path.exists(path_model):
        os.makedirs(path_model)

    with open(os.path.join(path_model, configfile), 'w') as f:
        json.dump(config, f, indent=2)
        
    name_model = path_model.split('/')[-2]

    file = open(os.path.join(path_model, bashname),"w")
    file.write("#!/bin/bash \n")
    file.write("#PBS -N mapping_test \n")
    file.write("#PBS -A rrp-355-aa \n")
    file.write("#PBS -l walltime=43200 \n")
    file.write("#PBS -l nodes=1:gpus=1 \n")
    file.write("cd $SCRATCH/maxime/axondeepseg/models/" + name_model + "/ \n")
    file.write("source /home/maxwab/tf11-py27/bin/activate \n")
    file.write("module load compilers/gcc/4.8.5 compilers/java/1.8 apps/buildtools compilers/swig apps/git apps/bazel/0.4.3 \n")
    file.write("module load cuda/7.5 \n")
    file.write("module load libs/cuDNN/5 \n")
    file.write("python ../../AxonDeepSeg/trainingforhelios.py -co ")
    file.write(str(configfile))
    file.write(" -t ") 
    file.write(str(path_trainingset))
    file.write(" -m ")
    file.write(str(path_model))
    if path_model_init:
        file.write(" -i ")
        file.write(str(path_model_init))
        
    file.close()

## ----------------------------------------------------------------------------------------------------------------
    
    
def generate_guilliminjob(bashname,configfile,config,path_trainingset,path_model,path_model_init = None):
    """Generate config file given a config dict. Generate the corresponding submission."""

    if not os.path.exists(path_model):
        os.makedirs(path_model)

    with open(os.path.join(path_model, configfile), 'w') as f:
        json.dump(config, f, indent=2)
        
    name_model = path_model.split('/')[-2]

    file = open(os.path.join(path_model, bashname),"w")
    file.write("#!/bin/bash \n")
    file.write("#PBS -N "+ name_model +" \n")
    file.write("#PBS -A rrp-355-aa \n")
    file.write("#PBS -l walltime=43200 \n")
    file.write("#PBS -l nodes=1:gpus=1 \n")
    file.write("cd /gs/project/rrp-355-aa/maxwab/axondeepseg/models/" + name_model + "/ \n")
    file.write("module load foss/2015b Python/2.7.12 \n")
    file.write("source ~/maxwab/tf11-py27/bin/activate \n")
    file.write("module load GCC/5.3.0-2.26 Bazel/0.4.4 CUDA/7.5.18 \n")
    file.write("module load Tensorflow/1.0.0-Python-2.7.12 \n")
    file.write("python ../../AxonDeepSeg/trainingforhelios.py -co ")
    file.write(str(configfile))
    file.write(" -t ") 
    file.write(str(path_trainingset))
    file.write(" -m ")
    file.write(str(path_model))
    if path_model_init:
        file.write(" -i ")
        file.write(str(path_model_init))
        
    file.close()