from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil


def download_model(destination = None):
    sem_release_version = 'r20211111'

    
    if destination is None:
        sem_destination = Path("AxonDeepSeg/models/default_SEM_model")
    else:
        sem_destination = destination / "default_SEM_model"

    url_sem_destination = "https://github.com/axondeepseg/default-SEM-model/archive/refs/tags/" + sem_release_version + ".zip"

    files_before = list(Path.cwd().iterdir())

    if (
        not download_data(url_sem_destination)
    ) == 1:
        print("Data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder names
    model_folders = list(set(files_after)-set(files_before))
    folder_name_SEM_model = ''.join([str(x) for x in model_folders if 'rat_axon' in str(x)])

    if sem_destination.exists():
        print('SEM model folder already existed - deleting old one')
        shutil.rmtree(str(sem_destination))
    
    shutil.move(Path("default_SEM_model-" + sem_release_version).joinpath("model_seg_rat_axon-myelin_sem"), str(sem_destination))

    # remove temporary folders
    shutil.rmtree(folder_name_SEM_model)

def main(argv=None):
    download_model()
