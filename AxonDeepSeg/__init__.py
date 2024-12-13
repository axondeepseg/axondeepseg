import os
import sys
import pygit2

__name__ = "AxonDeepSeg"
__version__ = "5.0.0"

# make sure repo directory (two directories up) is on path (to access config.py)
__repo_dir__ = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(__repo_dir__)

repo = pygit2.Repository(__repo_dir__)
__git_version__ = str(repo.head.target)
__version_string__ = f"{__name__} v{__version__} ({__git_version__})"