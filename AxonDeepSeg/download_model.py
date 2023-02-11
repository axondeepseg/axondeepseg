from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil
import argparse
from argparse import RawTextHelpFormatter

def download_model(destination = None):
    sem_release_version = 'r20211209v2'
    tem_release_version = 'r20211111v3'
    bf_release_version = 'r20211210'
    
    if destination is None:
        sem_destination = Path(__file__).parent / "model_seg_rat_axon-myelin_sem"
        tem_destination = Path(__file__).parent / "model_seg_mouse_axon-myelin_tem"
        bf_destination = Path(__file__).parent / "model_seg_rat_axon-myelin_bf"
    else:
        sem_destination = destination / "model_seg_rat_axon-myelin_sem"
        tem_destination = destination / "model_seg_mouse_axon-myelin_tem"
        bf_destination = destination / "model_seg_rat_axon-myelin_bf"

    model_working_directory = True if destination is not None and destination.name == Path.cwd().name else False # Flag to check if user wants to store model in the working directory or not
    
    url_sem_destination = "https://github.com/axondeepseg/default-SEM-model/archive/refs/tags/" + sem_release_version + ".zip"
    url_tem_destination = "https://github.com/axondeepseg/default-TEM-model/archive/refs/tags/" + tem_release_version + ".zip"
    url_bf_destination = "https://github.com/axondeepseg/default-BF-model/archive/refs/tags/" + bf_release_version + ".zip"

    if sem_destination.exists():                                                
        print('SEM model folder already existed - deleting old one.')
        shutil.rmtree(str(sem_destination))
    if tem_destination.exists():
        print('TEM model folder already existed - deleting old one.')
        shutil.rmtree(str(tem_destination))
    if bf_destination.exists():
        print('BF model folder already existed - deleting old one.')
        shutil.rmtree(str(bf_destination))

    files_before = list(Path.cwd().iterdir())

    if (
        not download_data(url_sem_destination) and not download_data(url_tem_destination) and not download_data(url_bf_destination) 
    ) == 1:
        print("Data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    if not model_working_directory:           # if destination folder is other than root directory
        shutil.move("model_seg_rat_axon-myelin_sem", str(sem_destination))
        shutil.move("model_seg_mouse_axon-myelin_tem", str(tem_destination))
        shutil.move("model_seg_rat_axon-myelin_bf", str(bf_destination))

    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder names
    folder_name_SEM_model = Path("default-SEM-model-" + sem_release_version)
    folder_name_TEM_model = Path("default-TEM-model-" + tem_release_version)
    folder_name_BF_model = Path("default-BF-model-" + bf_release_version)

    if sem_destination.exists():
        print('SEM model folder already existed - deleting old one')
        shutil.rmtree(str(sem_destination))

    if tem_destination.exists():
       print('TEM model folder already existed - deleting old one')
       shutil.rmtree(str(tem_destination))      

    if bf_destination.exists():
       print('BF model folder already existed - deleting old one')
       shutil.rmtree(str(bf_destination))

    shutil.move(folder_name_SEM_model.joinpath("model_seg_rat_axon-myelin_sem"), str(sem_destination))
    shutil.move(folder_name_TEM_model.joinpath("model_seg_mouse_axon-myelin_tem"), str(tem_destination))
    shutil.move(folder_name_BF_model.joinpath("model_seg_rat_axon-myelin_bf"), str(bf_destination))

    print("Models are stored in the directory: ", sem_destination.parent)

    # remove temporary folders
    shutil.rmtree(folder_name_SEM_model)
    shutil.rmtree(folder_name_TEM_model)
    shutil.rmtree(folder_name_BF_model)

def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    # Setting the argument for specifying the directory where you wish to download the models
    ap.add_argument("-p", "--path", required=False, help='Path where you wish to save your models', default = None)
    args = vars(ap.parse_args(argv))
    destination = args["path"]
    download_model(destination)
