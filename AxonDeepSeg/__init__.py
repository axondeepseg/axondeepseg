import os
import sys

# make sure repo directory (two directories up) is on path (to access config.py)
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

__version__ = "5.0.0"
