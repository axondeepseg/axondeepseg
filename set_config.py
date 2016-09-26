# USAGE
# python set_config.py

import argparse
import ConfigParser as cp

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path_axonseg", required=True, help="absolute path of the AxonSeg toolbox (matlab) - Used for myelin detection")
args = vars(ap.parse_args())

path_axonseg = args["path_axonseg"]

path_cfg = './AxonDeepSeg/data/config.cfg'
config = cp.RawConfigParser()
config.read(path_cfg)
config.set('paths', 'path_axonseg', path_axonseg )

config.write(open(path_cfg,'w'))



