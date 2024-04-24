from AxonDeepSeg.ads_utils import convert_path, download_data
from AxonDeepSeg.model_cards import MODELS
from pathlib import Path
import shutil
from loguru import logger
import sys
import argparse


def download_model(model='generalist', model_type='light', destination=None):
    
    if destination is None:
        model_destination = Path(f"AxonDeepSeg/models/{model}")
    else:
        model_destination = destination / model

    url_model_destination = MODELS[model]['weights'][model_type]
    if url_model_destination is None:
        print(f"Model not found.")
        sys.exit()

    files_before = list(Path.cwd().iterdir())
    if download_data(url_model_destination) == 0:
        print("Model downloaded and unzipped succesfully.")
    else:
        print("An error occured. The model was not downloaded.")
        sys.exit()
    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder name
    folder_name = list(set(files_after) - set(files_before))[0]

    if model_destination.exists():
        print('Model folder already existed - deleting old one')
        shutil.rmtree(str(model_destination))
    
    model_suffix = 'light' if model_type == 'light' else 'ensemble'
    model_name = f'{MODELS[model]["name"]}_{model_suffix}'
    shutil.move(folder_name.joinpath(model_name), str(model_destination))

    # remove temporary folder
    shutil.rmtree(folder_name)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model-name",
        required=False,
        help="Model to download. Default: generalist",
        default='generalist',
        type=str,
    )
    ap.add_argument(
        "-t", "--model-type",
        required=False,
        help="Model type to download. Default: light",
        default='light',
        type=str,
    )
    args = vars(ap.parse_args())

    download_model(args["model_name"], args["model_type"])

if __name__ == "__main__":
    with logger.catch():
        main()