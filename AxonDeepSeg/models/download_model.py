from AxonDeepSeg.ads_utils import *
from pathlib import Path
import shutil


def download_model(destination = None):
    if destination is None:
        sem_destination = Path("AxonDeepSeg/models/default_SEM_model")
        tem_destination = Path("AxonDeepSeg/models/default_TEM_model")
    else:
        destination = convert_path(destination)
        sem_destination = destination / "default_SEM_model"
        tem_destination = destination / "default_TEM_model"

    url_TEM_model = "https://osf.io/2hcfv/?action=download&version=5"  # URL of TEM model hosted on OSF storage with the appropriate versioning on OSF
    url_SEM_model = "https://osf.io/sv7u2/?action=download&version=5"  # URL of SEM model hosted on OSF storage with the appropriate versioning on OSF

    if (
        not download_data(url_TEM_model) and not download_data(url_SEM_model)
    ) == 1:
        print("Data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    if sem_destination.exists():
        print('SEM model folder already existed - deleting old one.')
        shutil.rmtree(str(sem_destination))
    if tem_destination.exists():
        print('TEM model folder already existed - deleting old one.')
        shutil.rmtree(str(tem_destination))

    shutil.move("default_SEM_model", str(sem_destination))
    shutil.move("default_TEM_model", str(tem_destination))


def main(argv=None):
    download_model()
