from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil
import requests


def download_model(destination = None):
    if destination is None:
        sem_destination = Path("AxonDeepSeg/models/default_SEM_model")
        tem_destination = Path("AxonDeepSeg/models/default_TEM_model")
        model_seg_pns_bf_destination = Path("AxonDeepSeg/models/model_seg_pns_bf")
    else:
        destination = convert_path(destination)
        sem_destination = destination / "default_SEM_model"
        tem_destination = destination / "default_TEM_model"
        model_seg_pns_bf_destination = destination / "model_seg_pns_bf"

    # retrieve latest release identifier from github API
    response_TEM = requests.get("https://api.github.com/repos/axondeepseg/default-TEM-model/releases/latest")
    folder_name_TEM_model = Path("default-TEM-model-"+str(response_TEM.json()['tag_name']))
    response_SEM = requests.get("https://api.github.com/repos/axondeepseg/default-SEM-model/releases/latest")
    folder_name_SEM_model = Path("default-SEM-model-"+str(response_SEM.json()['tag_name']))
    response_OM = requests.get("https://api.github.com/repos/axondeepseg/model-seg-pns-bf/releases/latest")
    folder_name_OM_model = Path("model-seg-pns-bf-"+str(response_OM.json()['tag_name']))

    url_TEM_model = "https://github.com/axondeepseg/default-TEM-model/archive/refs/tags/"+str(response_TEM.json()['tag_name'])+".zip" 
    url_SEM_model = "https://github.com/axondeepseg/default-SEM-model/archive/refs/tags/"+str(response_SEM.json()['tag_name'])+".zip" 
    url_model_seg_pns_bf = "https://github.com/axondeepseg/model-seg-pns-bf/archive/refs/tags/"+str(response_OM.json()['tag_name'])+".zip"

    if (
        not download_data(url_TEM_model) and not download_data(url_SEM_model) and not download_data(url_model_seg_pns_bf)
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
    if model_seg_pns_bf_destination.exists():
        print('Bright Field Optical Microscopy model folder already existed - deleting old one')
        shutil.rmtree(str(model_seg_pns_bf_destination))

    shutil.move(folder_name_SEM_model.joinpath("default_SEM_model"), str(sem_destination))
    shutil.move(folder_name_TEM_model.joinpath("default_TEM_model"), str(tem_destination))
    shutil.move(folder_name_OM_model.joinpath("model_seg_pns_bf"), str(model_seg_pns_bf_destination))

    # remove temporary folders
    shutil.rmtree(folder_name_TEM_model)
    shutil.rmtree(folder_name_SEM_model)
    shutil.rmtree(folder_name_OM_model)

def main(argv=None):
    download_model()
