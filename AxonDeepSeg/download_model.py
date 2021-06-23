from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil


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

    url_TEM_model = "https://github.com/axondeepseg/default-TEM-model/archive/refs/tags/r20210615.zip" 
    url_SEM_model = "https://github.com/axondeepseg/default-SEM-model/archive/refs/tags/r20210615.zip" 
    url_model_seg_pns_bf = "https://github.com/axondeepseg/model-seg-pns-bf/archive/refs/tags/r20210615.zip"

    files_before = list(Path.cwd().iterdir())

    if (
        not download_data(url_TEM_model) and not download_data(url_SEM_model) and not download_data(url_model_seg_pns_bf)
    ) == 1:
        print("Data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder names
    model_folders = list(set(files_after)-set(files_before))
    folder_name_TEM_model = ''.join([str(x) for x in model_folders if 'TEM' in str(x)])
    folder_name_SEM_model = ''.join([str(x) for x in model_folders if 'SEM' in str(x)])
    folder_name_OM_model = ''.join([str(x) for x in model_folders if 'pns-bf' in str(x)])

    if sem_destination.exists():
        print('SEM model folder already existed - deleting old one.')
        shutil.rmtree(str(sem_destination))
    if tem_destination.exists():
        print('TEM model folder already existed - deleting old one.')
        shutil.rmtree(str(tem_destination))
    if model_seg_pns_bf_destination.exists():
        print('Bright Field Optical Microscopy model folder already existed - deleting old one')
        shutil.rmtree(str(model_seg_pns_bf_destination))

    shutil.move(Path(folder_name_SEM_model).joinpath("default_SEM_model"), str(sem_destination))
    shutil.move(Path(folder_name_TEM_model).joinpath("default_TEM_model"), str(tem_destination))
    shutil.move(Path(folder_name_OM_model).joinpath("model_seg_pns_bf"), str(model_seg_pns_bf_destination))

    # remove temporary folders
    shutil.rmtree(folder_name_TEM_model)
    shutil.rmtree(folder_name_SEM_model)
    shutil.rmtree(folder_name_OM_model)

def main(argv=None):
    download_model()
