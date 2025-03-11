import AxonDeepSeg
from AxonDeepSeg.ads_utils import convert_path, download_data
from AxonDeepSeg.model_cards import MODELS
from pathlib import Path
import shutil
from loguru import logger
import sys
import argparse
import pprint

# exit codes
SUCCESS, MODEL_NOT_FOUND, DOWNLOAD_ERROR = 0, 1, 2

def download_model(model='generalist', model_type='light', destination=None, overwrite=True):
    '''
    Download a model for AxonDeepSeg.
    Parameters
    ----------
    model : str, optional
        Name of the model, by default 'generalist'. 
    model_type :  Literal['light', 'ensemble'], optional
        Type of model, by default 'light'. 
    destination : str, optional
        Directory to download the model to. Default: None.
    '''
    model_suffix = 'light' if model_type == 'light' else 'ensemble'
    full_model_name = f'{MODELS[model]["name"]}_{model_suffix}'

    if destination is None:
        package_dir = Path(AxonDeepSeg.__file__).parent  # Get AxonDeepSeg installation path
        model_destination = package_dir / "models" / full_model_name
        print('Downloading model to default location: {}'.format(model_destination))
    else:
        destination = Path(destination)
        model_destination = destination / full_model_name
    if model_destination.exists() and overwrite == False:
        logger.info("Overwrite set to False - not deleting old model.")
        return model_destination
    
    url_model_destination = MODELS[model]['weights'][model_type]
    if url_model_destination is None:
        logger.error('Model not found.')
        sys.exit(MODEL_NOT_FOUND)

    files_before = list(Path.cwd().iterdir())
    if download_data(url_model_destination) == 0:
        logger.info("Model downloaded and unzipped succesfully.")
    else:
        logger.error("An error occured. The model was not downloaded.")
        sys.exit(DOWNLOAD_ERROR)
    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder name
    folder_name = list(set(files_after) - set(files_before))[0]
    output_dir = model_destination.resolve()

    if model_destination.exists():
        logger.info("Model folder already existed - deleting old one")
        shutil.rmtree(str(model_destination))

    shutil.move(folder_name, str(model_destination))

    return output_dir

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
    ap.add_argument(
        "-l", "--list",
        required=False,
        help="List all available models for download",
        default=False,
        action='store_true',
    )
    ap.add_argument(
        "-d", "--dir",
        required=False,
        help="Directory to download the model to. Default: AxonDeepSeg/models",
        default = None,
    )
    args = vars(ap.parse_args(argv))

    if args["list"]:
        logger.info("Printing available models:")
        for model in MODELS:
            logger.info(model)
            model_details = {
                "MODEL NAME": MODELS[model]['name'],
                "NUMBER OF CLASSES": MODELS[model]['n_classes'],
                "OVERVIEW": MODELS[model]['model-info'],
                "TRAINING DATA": MODELS[model]['training-data'],
            }
            pprint.pprint(model_details)
        sys.exit(SUCCESS)
    else:
        download_model(args["model_name"], args["model_type"], args["dir"], overwrite=True)

if __name__ == "__main__":
    with logger.catch():
        main()