# -*- coding: utf-8 -*-

# Script to place in the same folder as train_network.py

import sys
import os
import json
import AxonDeepSeg.ads_utils


def compute_training(configfile, path_trainingset, path_model, path_model_init = None, gpu_per = 1.0):

    os.chdir(sys.path[0]) # Necessary to fix the directory we are working in
    with open(path_model / configfile, 'r') as fd:
        config_network = json.loads(fd.read())

    #if not os.path.exists(path_model): # Already created before. But can be useful if we want to create models with date of launch.
    #    os.makedirs(path_model)

    #with open(path_model + configname, 'w') as f:
    #    json.dump(config, f, indent=2)

    #with open(path_model + filename, 'r') as fd:
    #    config_network = json.loads(fd.read())

    # Training
    from AxonDeepSeg.train_network import train_model
    train_model(path_trainingset, path_model, config_network, gpu_per=gpu_per)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-co", "--configfile", required=True, help="")
    ap.add_argument("-t", "--path_trainingset", required=True, help="")
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-i", "--path_model_init", required=False, default = None, help="")
    ap.add_argument("-g", "--gpu_per", required=False, default = 1.0, help="")

    args = vars(ap.parse_args())
    configfile = str(args["configfile"])
    path_trainingset = str(args["path_trainingset"])
    path_model = str(args["path_model"])
    path_model_init = str(args["path_model_init"])
    gpu_per = float(args["gpu_per"])

    compute_training(configfile, path_trainingset, path_model, path_model_init, gpu_per = gpu_per)

if __name__ == '__main__':
    main()