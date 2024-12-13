import os
import sys
import git

# make sure repo directory (two directories up) is on path (to access config.py)
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

__name__ = "AxonDeepSeg"
__version__ = "5.0.0"
__git_version__ = git.Repo(search_parent_directories=True).head.object.hexsha
__version_string__ = f"{__name__} v{__version__} ({__git_version__})"