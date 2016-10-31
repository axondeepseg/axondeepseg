'''
Set the config variable.
'''

import ConfigParser as cp
import os

config = cp.RawConfigParser()
config.read(os.path.dirname(__file__)+'/data/config.cfg')


# Global variables
path_axonseg = config.get("paths", "path_axonseg")
path_matlab = config.get("paths", "path_matlab")
general_pixel_size = float(config.get("variables", "general_pixel_size"))

