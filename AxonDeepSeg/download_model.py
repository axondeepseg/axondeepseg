from AxonDeepSeg.ads_utils import convert_path, download_data
from AxonDeepSeg.model_cards import MODELS
from pathlib import Path
import shutil


def download_model(model='generalist', model_type='light', destination=None):
    
    if destination is None:
        model_destination = Path(f"AxonDeepSeg/models/{model}")
    else:
        model_destination = destination / model

    url_model_destination = MODELS[model]['weights'][model_type]

    files_before = list(Path.cwd().iterdir())
    if not download_data(url_model_destination):
        print("Model downloaded and unzipped succesfully.")
    else:
        print("ERROR: Model was not succesfully downloaded and unzipped- "
              "please check your link and filename and try again.")
    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder name
    folder_name = list(set(files_after) - set(files_before))[0]

    if model_destination.exists():
        print('Model folder already existed - deleting old one')
        shutil.rmtree(str(model_destination))
    
    model_suffix = 'light' if model_type == 'light' else 'ensemble'
    model_name = f'{MODELS[model]['name']}_{model_suffix}'
    shutil.move(folder_name.joinpath(model_name), str(model_destination)

    # remove temporary folder
    shutil.rmtree(folder_name)

def main(argv=None):
    download_model()
