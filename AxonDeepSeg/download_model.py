from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil


def download_model(destination = None):
    if destination is None:
        model_seg_rat_axon_myelin_sem = Path("AxonDeepSeg/models/model_seg_rat_axon-myelin_sem")
    else:
        model_seg_rat_axon_myelin_sem = destination / "model_seg_rat_axon-myelin_sem"

    url_model_seg_rat_axon_myelin_sem = "https://github.com/axondeepseg/model_seg_rat_axon-myelin_sem/archive/refs/tags/r20210806.zip"

    files_before = list(Path.cwd().iterdir())

    if (
        not download_data(url_model_seg_rat_axon_myelin_sem)
    ) == 1:
        print("Data downloaded and unzipped succesfully.")
    else:
        print(
            "ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again."
        )

    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder names
    model_folders = list(set(files_after)-set(files_before))
    folder_name_SEM_ivadomed_model = ''.join([str(x) for x in model_folders if 'rat_axon' in str(x)])

    if model_seg_rat_axon_myelin_sem.exists():
        print('IVADOMED SEM model folder already existed - deleting old one')
        shutil.rmtree(str(model_seg_rat_axon_myelin_sem))
        
    shutil.move(Path(folder_name_SEM_ivadomed_model).joinpath("model_seg_rat_axon-myelin_sem"), str(model_seg_rat_axon_myelin_sem))

    # remove temporary folders
    shutil.rmtree(folder_name_SEM_ivadomed_model)

def main(argv=None):
    download_model()
