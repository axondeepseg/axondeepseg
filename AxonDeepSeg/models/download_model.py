from AxonDeepSeg.ads_utils import *
from pathlib import Path
import shutil
import argparse
from argparse import RawTextHelpFormatter

def download_model(destination = None):
    root_directory = False             #Flag to check if the destination folder is the root directory 
    if destination is None:
        sem_destination = Path("AxonDeepSeg/models/default_SEM_model")
        tem_destination = Path("AxonDeepSeg/models/default_TEM_model")
    else:
        destination = convert_path(destination)
        sem_destination = destination / "default_SEM_model"
        tem_destination = destination / "default_TEM_model"
    print("SEM destination is", sem_destination)
    print("TEM destination is", tem_destination)
    url_TEM_model = "https://osf.io/2hcfv/?action=download&version=5"  # URL of TEM model hosted on OSF storage with the appropriate versioning on OSF
    url_SEM_model = "https://osf.io/sv7u2/?action=download&version=5"  # URL of SEM model hosted on OSF storage with the appropriate versioning on OSF

    if destination is not None and destination.name =="axondeepseg":     #Deletes existing models if present in the root directory 
        root_directory = True
        if sem_destination.exists():
            print('SEM model folder already existed - deleting old one.')
            shutil.rmtree(str(sem_destination))
        if tem_destination.exists():
            print('TEM model folder already existed - deleting old one.')
            shutil.rmtree(str(tem_destination))

    if (
        not download_data(url_TEM_model) and not download_data(url_SEM_model)
    ) == 1:
        print("Data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    if not root_directory:           # if destination folder is other than root directory
        if sem_destination.exists():
            print('SEM model folder already existed - deleting old one.')
            shutil.rmtree(str(sem_destination))
        if tem_destination.exists():
            print('TEM model folder already existed - deleting old one.')
            shutil.rmtree(str(tem_destination))

        shutil.move("default_SEM_model", str(sem_destination))
        shutil.move("default_TEM_model", str(tem_destination))

    print("Models are stored in the directory: ", sem_destination.parent)
    

def main(argv=None):

    ap = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    # Setting the argument for specifying the directory where you wish to download the models
    ap.add_argument("-p", "--path", required=False, help='Path where you wish to save your models', default = None)
    args = vars(ap.parse_args(argv))
    destination = args["path"]
    download_model(destination)
