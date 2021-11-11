from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil


def download_model(destination = None):
    sem_release_version = 'r20211111v2'
    tem_release_version = 'r20211111v2'
    
    if destination is None:
        sem_destination = Path("AxonDeepSeg/models/model_seg_rat_axon-myelin_sem")
        tem_destination = Path("AxonDeepSeg/models/model_seg_mouse_axon-myelin_tem")
    else:
        sem_destination = destination / "model_seg_rat_axon-myelin_sem"
        tem_destination = destination / "model_seg_mouse_axon-myelin_tem"

    url_sem_destination = "https://github.com/axondeepseg/default-SEM-model/archive/refs/tags/" + sem_release_version + ".zip"
    url_tem_destination = "https://github.com/axondeepseg/default-TEM-model/archive/refs/tags/" + tem_release_version + ".zip"

    files_before = list(Path.cwd().iterdir())

    if (
        not download_data(url_sem_destination) and not download_data(url_tem_destination)
    ) == 1:
        print("Data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder names
    folder_name_SEM_model = Path("default-SEM-model-" + sem_release_version)
    folder_name_TEM_model = Path("default-TEM-model-" + tem_release_version)

    if sem_destination.exists():
        print('SEM model folder already existed - deleting old one')
        shutil.rmtree(str(sem_destination))

    if tem_destination.exists():
       print('TEM model folder already existed - deleting old one')
       shutil.rmtree(str(tem_destination))      

    shutil.move(folder_name_SEM_model.joinpath("model_seg_rat_axon-myelin_sem"), str(sem_destination))
    shutil.move(folder_name_TEM_model.joinpath("model_seg_mouse_axon-myelin_tem"), str(tem_destination))

    # remove temporary folders
    shutil.rmtree(folder_name_SEM_model)
    shutil.rmtree(folder_name_TEM_model)

def main(argv=None):
    download_model()
