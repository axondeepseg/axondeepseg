  
# USAGE
# python set_config.py

import argparse
import configparser as cp

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path_axonseg", required=True, help="absolute path of the AxonSeg toolbox (matlab) - Used for myelin detection")
ap.add_argument("-m", "--path_matlab", required=True, help="absolute path of matlab (to find it run matlabroot in matlab terminal)")
args = vars(ap.parse_args())

path_axonseg = args["path_axonseg"]
path_matlab = args["path_matlab"]

path_cfg = './AxonDeepSeg/data/config.cfg'
config = cp.RawConfigParser()
config.read(path_cfg)
config.set('paths', 'path_axonseg', path_axonseg)
config.set('paths', 'path_matlab', path_matlab)

config.write(open(path_cfg, 'w'))