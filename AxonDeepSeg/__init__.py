import os
import sys

__name__ = "AxonDeepSeg"

version_file = os.path.abspath(os.path.join(__file__, os.pardir, "version.txt"))
with open(version_file, 'r') as f:
    __version__ = f.read().rstrip()

# make sure repo directory (two directories up) is on path (to access config.py)
__repo_dir__ = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(__repo_dir__)
