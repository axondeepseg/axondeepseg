import os
import sys

__name__ = "AxonDeepSeg"

version_file = os.path.abspath(os.path.join(__file__, os.pardir, "version.txt"))
with open(version_file, 'r') as f:
    __version__ = f.read().rstrip()

# make sure repo directory (two directories up) is on path (to access config.py)
__repo_dir__ = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(__repo_dir__)

# verify that the repo was downloaded with `git clone`
if os.path.exists(os.path.join(__repo_dir__, ".git")):
    import subprocess
    __git_sha1__ = subprocess.check_output(["git", "-C", __repo_dir__, "rev-parse", "HEAD"]).decode("utf-8").rstrip()
else:
    __git_sha1__ = "unknown commit"    
__version_string__ = f"AxonDeepSeg v{__version__} - {__git_sha1__}"