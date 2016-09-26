'''
Set the config variable.
'''

import ConfigParser as cp
import os

config = cp.RawConfigParser()
path = os.path.abspath('../data/config.cfg')
config.read(path)

path_axonseg = config.get("paths", "path_axonseg")
general_pixel_size = float(config.get("variables", "general_pixel_size"))




