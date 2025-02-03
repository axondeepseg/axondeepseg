import os
import sys
import pygit2

__name__ = "ads_base"

version_file = os.path.abspath(os.path.join(__file__, os.pardir, "version.txt"))
with open(version_file, 'r') as f:
    __version__ = f.read().rstrip()

# make sure repo directory (two directories up) is on path (to access config.py)
#__repo_dir__ = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
#sys.path.append(__repo_dir__)

#repo = pygit2.Repository(__repo_dir__)
#__git_version__ = str(repo.head.target)
#__git_name__ = repo.head.shorthand
#__version_string__ = f"{__name__} v{__version__} ({__git_name__}: {__git_version__})"
