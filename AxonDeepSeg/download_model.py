import AxonDeepSeg
from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil
from loguru import logger
import sys
import argparse
import yaml
import textwrap

# exit codes
SUCCESS_EXIT_CODE = 0
MODEL_NOT_FOUND_CODE = 1
DOWNLOAD_ERROR_CODE = 2

MODEL_CARDS_PATH = Path(__file__).parent / 'model_cards.yaml'

def download_model(model_name='generalist', destination=None, overwrite=True):
    '''
    Download a model for AxonDeepSeg.
    Parameters
    ----------
    model_name : str, optional
        Name of the model, by default 'generalist'. 
    model_type :  Literal['light', 'ensemble'], optional
        Type of model, by default 'light'. 
    destination : str, optional
        Directory to download the model to. Default: None.
    '''
    models = get_model_cards(Path(__file__).parent / 'model_cards.yaml')
    if model_name not in models.keys():
        logger.error('Model not found.')
        sys.exit(MODEL_NOT_FOUND_CODE)

    # default to single_fold model if available (lighter and faster)
    if models[model_name]['weights']['single_fold'] is not None:
        model_suffix = 'light'
        url_model_destination = models[model_name]['weights']['single_fold']
    elif models[model_name]['weights']['ensemble'] is not None:
        model_suffix = 'ensemble'
        url_model_destination = models[model_name]['weights']['ensemble']

    full_model_name = f'{models[model_name]["full_name"]}_{model_suffix}'
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

    files_before = list(Path.cwd().iterdir())
    if download_data(url_model_destination) == 0:
        logger.info("Model downloaded and unzipped succesfully.")
    else:
        logger.error("An error occured. The model was not downloaded.")
        sys.exit(DOWNLOAD_ERROR_CODE)
    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder name
    folder_name = list(set(files_after) - set(files_before))[0]
    output_dir = model_destination.resolve()

    if model_destination.exists():
        logger.info("Model folder already existed - deleting old one")
        shutil.rmtree(str(model_destination))

    shutil.move(folder_name, str(model_destination))

    return output_dir

def format_pixel_size(pixel_size) -> str:
    '''
    Format pixel size value(s) for display.
    If a list, show all individual values.
    If a single value, show it directly.
    '''
    if isinstance(pixel_size, list):
        values_str = ', '.join(str(v) for v in pixel_size)
        return f'{values_str} µm/px'
    return f'{pixel_size} µm/px'


def print_available_models(model_dict: dict):
    '''
    Print all available models for download.
    '''
    logger.info("Printing available models:")
    for model in model_dict:
        pixel_size = model_dict[model].get('pixel_size')
        pixel_size_str = format_pixel_size(pixel_size) if pixel_size is not None else 'N/A'
        to_print = [
            ["Model name", model],
            ["Nb of classes", model_dict[model]['n_classes']],
            ["Pixel size", pixel_size_str],
            ["Model info", model_dict[model]['model-info']],
            ["Training data", model_dict[model]['training-data']],
        ]
        print("\n")
        for label, content in to_print:
            print(label, '\t', textwrap.fill(str(content), width=90).replace('\n', '\n\t\t '))
            
def get_model_cards(model_list_path=MODEL_CARDS_PATH) -> dict:
    '''
    Load the model list from a YAML file.
    '''
    with open(model_list_path, 'r') as f:
        model_dict = yaml.safe_load(f)
    return model_dict

def get_fullnames_of_downloadable_models() -> list:
    '''
    Get the full names of all downloadable models.
    '''
    model_cards = get_model_cards(MODEL_CARDS_PATH)
    downloadable_models = []
    for model_name, model_info in model_cards.items():
        if model_info['weights']['single_fold'] is not None:
            downloadable_models.append(model_info['full_name'] + '_light')
    return downloadable_models


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

    model_cards = get_model_cards()

    if args["list"]:
        print_available_models(model_cards)
        sys.exit(SUCCESS_EXIT_CODE)
    else:
        download_model(args["model_name"], args["dir"], overwrite=True)

if __name__ == "__main__":
    with logger.catch():
        main()