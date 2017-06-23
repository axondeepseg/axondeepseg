import json
import os
import numpy as np
from AxonDeepSeg.config_tools import *


## ----------------------------------------------------------------------------------------------------------------

def generate_heliosjob(path_project, path_venv, bashname, configfile, config, path_trainingset, path_model, path_model_init = None, walltime=43200):
    """Generate config file given a config dict. Generate the corresponding submission."""

    if not os.path.exists(path_model):
        os.makedirs(path_model)

    with open(os.path.join(path_model, configfile), 'w') as f:
        json.dump(config, f, indent=2)
        
    name_model = path_model.split('/')[-1]

    file = open(os.path.join(path_model, bashname),"w")
    file.write("#!/bin/bash \n")
    file.write("#PBS -N mapping_test \n")
    file.write("#PBS -A rrp-355-aa \n")
    file.write("#PBS -l walltime="+str(walltime)+" \n")
    file.write("#PBS -l nodes=1:gpus=1 \n")
    file.write("cd $SCRATCH/"+path_project+ "/axondeepseg/models/" + name_model + "/ \n")
    file.write("source "+path_venv+"/bin/activate \n")
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
    
    
def generate_guilliminjob(path_project, path_venv, bashname, configfile, config, path_trainingset, path_model,path_model_init = None, walltime=43200):
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
    file.write("#PBS -l walltime="+str(walltime)+" \n")
    file.write("#PBS -l nodes=1:gpus=1 \n")
    file.write("cd "+path_project+"/axondeepseg/models/" + name_model + "/ \n")
    file.write("module load foss/2015b Python/2.7.12 \n")
    file.write("source "+path_venv+"/bin/activate \n")
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