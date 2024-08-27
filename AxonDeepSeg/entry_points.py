# Helper entry points to make things easier for users
import os
import sys
import subprocess
from pathlib import Path

ADS_DIR = (Path(os.environ['ADS_DIR']) if 'ADS_DIR' in os.environ
           else Path(__file__).parent.parent.absolute())
SCRIPTS_DIRNAME = "Scripts" if sys.platform == 'win32' else "bin"
VENV_DIR = ADS_DIR / "ads_conda" / "envs" / "venv_ads"


def activate_conda_env():
    print("Activating ADS virtual environment...")
    subprocess.call([f"{ADS_DIR / 'ads_conda' / SCRIPTS_DIRNAME / 'activate'}", f"{VENV_DIR}"])


def run_napari():
    print("Launching Napari...")
    subprocess.call([f"{VENV_DIR / SCRIPTS_DIRNAME / 'napari'}"])
