'''
Set the config variable.
'''

import ConfigParser as cp
import os
import AxonDeepSeg

config = cp.RawConfigParser()
config.read(os.path.dirname(AxonDeepSeg.__file__)+'/data/config.cfg')
path_axonseg = config.get("paths", "path_axonseg")
general_pixel_size = float(config.get("variables", "general_pixel_size"))

