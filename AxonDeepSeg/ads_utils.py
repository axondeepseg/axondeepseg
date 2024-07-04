"""
AxonDeepSeg utilities module.
"""

import os
import sys
from pathlib import Path
import configparser
from distutils.util import strtobool
import cgi
import tempfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from tqdm import tqdm
import raven
import imageio
import numpy as np
from loguru import logger

from config import valid_extensions


DEFAULT_CONFIGFILE = "axondeepseg.cfg"

# raven function override - do not modify unless needed if raven version is
# changed to a version other than 6.8.0.
# See https://github.com/getsentry/raven-python/blob/master/raven/transport/threaded.py
# -- Start function override -- #
def _main_thread_terminated(self):
    self._lock.acquire()
    try:
        if not self.is_alive():
            # thread not started or already stopped - nothing to do
            return

        # wake the processing thread up
        self._queue.put_nowait(self._terminator)

        timeout = self.options['shutdown_timeout']

        # wait briefly, initially
        initial_timeout = 0.1
        if timeout < initial_timeout:
            initial_timeout = timeout

        if not self._timed_queue_join(initial_timeout):
            # if that didn't work, wait a bit longer
            # NB that size is an approximation, because other threads may
            # add or remove items
            size = self._queue.qsize()

            print(("Sentry is attempting to send %i pending error messages"
                  % size))
            print(("Waiting up to %s seconds" % timeout))

            if os.name == 'nt':
                print("Press Ctrl-Break to quit")
            else:
                print("Press Ctrl-C to quit")

            # -- Function override statement --#
            config_path = get_config_path()
            print(("Note: you can opt out of Sentry reporting by changing the "
                   "value of bugTracking to 0 in the "
                   "file {}".format(config_path)))
            # -- EO Function override statement --#

            self._timed_queue_join(timeout - initial_timeout)

        self._thread = None

    finally:
        self._lock.release()


raven.transport.threaded.AsyncWorker.main_thread_terminated = _main_thread_terminated
# -- End function override -- #

def config_setup():

    config_path = get_config_path()

    if 'pytest' in sys.modules:
        bugTracking = bool(0)
    else:
        print ("To improve user experience and fix bugs, the ADS development team "
               "is using a report system to automatically receive crash reports "
               "and errors from users. These reports are anonymous.")

        bugTracking = strtobool(
            input("Do you agree to help us improve ADS? [y]es/[n]o:")
            )

    if bugTracking:
        print(("Note: you can opt out of Sentry reporting by changing the "
               "value of bugTracking from 1 to 0 in the "
               "file {}".format(config_path)))

    config = configparser.ConfigParser()
    config['Global'] = {
        'bugTracking': bugTracking
    }

    with open(config_path, 'w') as configFile:
        config.write(configFile)

    print("Configuration saved successfully !")

def get_config_path():
    """Get the full path of the AxonDeepSeg configuration file.
    :return: String with the full path to the ADS config file.
    """
    return Path.home() / DEFAULT_CONFIGFILE


def read_config():
    """Read the system configuration file.
    :return: a dict with the configuration parameters.
    """

    config_path = get_config_path()

    if not config_path.exists():
        raise IOError("Could not find configuration file.")

    config = configparser.ConfigParser()
    config.read(str(config_path))

    return config


def init_ads():
    """ Initialize ads for typical terminal usage
    :return:
    """

    config_path = get_config_path()

    if not config_path.is_file():
        config_setup()
    else:
        pass

    config = read_config()

    init_error_client(config.get('Global','bugTracking'))


def init_error_client(bugTracking):
    """ Send traceback to neuropoly servers
    :return:
    """

    if strtobool(bugTracking):

        try:

            client = raven.Client(
                        "https://e04a130541c64bc9a64939672f19ad52@sentry.io/1238683",
                        processors=(
                            'raven.processors.RemoveStackLocalsProcessor',
                            'raven.processors.SanitizePasswordsProcessor')
                        )

            traceback_to_server(client)

        except:
            print("Unexpected error: bug tracking may not be functionning.")


def traceback_to_server(client):
    """
        Send all traceback children of Exception to sentry
    """

    def excepthook(exctype, value, traceback):
        if issubclass(exctype, Exception):
            client.captureException(exc_info=(exctype, value, traceback))
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = excepthook

def download_data(url_data):
    """ Downloads and extracts zip files from the web.
    :return: 0 - Success, 1 - Encountered an exception.
    """
    # Download
    try:
        print('Trying URL: %s' % url_data)
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=retry))
        response = session.get(url_data, stream=True)

        if "Content-Disposition" in response.headers:
            _, content = cgi.parse_header(response.headers['Content-Disposition'])
            zip_filename = content["filename"]
        else:
            print("Unexpected: link doesn't provide a filename")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / zip_filename
            with open(tmp_path, 'wb') as tmp_file:
                total = int(response.headers.get('content-length', 1))
                tqdm_bar = tqdm(total=total, unit='B', unit_scale=True, desc="Downloading", ascii=True)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        dl_chunk = len(chunk)
                        tqdm_bar.update(dl_chunk)
                tqdm_bar.close()
            # Unzip
            print("Unzip...")
            try:
                with zipfile.ZipFile(str(tmp_path)) as zf:
                    zf.extractall(".")
            except (zipfile.BadZipfile):
                print('ERROR: ZIP package corrupted. Please try downloading again.')
                return 1
            print("--> Folder created: " + str(Path.cwd() / Path(zip_filename).stem))
    except Exception as e:
        print("ERROR: %s" % e)
        return 1
    return 0

def convert_path(object_path):
    """
    Convert path or list of paths to Path() objects.
    If None type, returns None.

    :param object_path: string, Path() object, None, or a list of these.
    :return: Path() object, None, or a list of these.
    """
    if isinstance(object_path, list):
        path_list = []
        for path_iter in object_path:
            if isinstance(path_iter, Path):
                path_list.append(path_iter.absolute())
            elif isinstance(path_iter, str):
                path_list.append(Path(path_iter).absolute())
            elif path_iter == None:
                path_list.append(None)
            else:
                raise TypeError('Paths, folder names, and filenames must be either strings or pathlib.Path objects. object_path was type: ' + str(type(object_path)))
        return path_list
    else:
        if isinstance(object_path, Path):
            return object_path.absolute()
        elif isinstance(object_path, str):
            return Path(object_path).absolute()
        elif object_path == None:
            return None
        else:
            raise TypeError('Paths, folder names, and filenames must be either strings or pathlib.Path objects. object_path was type: ' + str(type(object_path)))

def imread(filename):
    """ Read image and convert it to desired bitdepth without truncation.
    """

    # Convert to Path
    filename = Path(filename)

    # Get list of all suffixes in file, join them into a string, and then 
    # lowercase to set the file extention to check against valid extension.
    file_ext = get_file_extension(filename)

    if (not file_ext) or ("ome" in file_ext):
            raise IOError(f"The input file extension '{file_ext}' of '{Path(filename).name}' is not "
                               f"supported. AxonDeepSeg supports the following "
                               f"file extensions:  '.png', '.tif', '.tiff', '.jpg' and '.jpeg'.")

    # Load image
    if str(file_ext) in ['.tif', '.tiff']:
        raw_img = imageio.v2.imread(filename, format='tiff-pil')
        if len(raw_img.shape) > 2:
            raw_img = imageio.v2.imread(filename, format='tiff-pil', mode='L')
    else:
        raw_img = imageio.v2.imread(filename)
        if len(raw_img.shape) > 2:
            raw_img = imageio.v2.imread(filename, mode='L')
    img = imageio.core.image_as_uint(raw_img, bitdepth=8)
    return img

def imwrite(filename, img, format='png'):
    """ Write image.
    """
    # check datatype:
    if img.dtype == 'float64':
        img = img.astype(np.uint8)
    imageio.imwrite(filename, img, format=format)

def extract_axon_and_myelin_masks_from_image_data(image_data):
    """
    Returns the binary axon and myelin masks from the image data.
    :param image_data: the image data that contains the 8-bit greyscale data, with over 200 (usually 255 if following
    the ADS convention) being axons, 100 to 200 (usually 127 if following the ADS convention) being myelin
    and 0 being background
    :return axon_mask: the binairy axon mask
    :return myelin_mask: the binary myelin mask
    """
    image_data_array = np.array(image_data)
    axon_mask = image_data_array > 200
    myelin_mask = (image_data_array > 100) & (image_data_array < 200)

    axon_mask = axon_mask.astype(np.uint8)
    myelin_mask = myelin_mask.astype(np.uint8)

    return axon_mask, myelin_mask
    
def get_existing_models_list():
    """
    This method returns a list containing the name of the existing models located under AxonDeepSeg/models
    :return: list containing the name of the existing models
    :rtype: list of strings
    """
    ADS_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(ADS_path, "models")
    models_list = next(os.walk(models_path))[1]
    if "__pycache__" in models_list:
        models_list.remove("__pycache__")
    return models_list

def get_file_extension(filename):
    """ Get file extension if it is supported
    Args:
        filename (str): Path of the file.
    Returns:
        str: File extension
    """
    # Find the first match from the list of supported file extensions
    extension = next((ext for ext in valid_extensions if str(filename).lower().endswith(ext)), None)
    return extension

def get_imshape(filename: str):
    """Get the shape of an image (HWC format) without reading its data.
    """
    shape = imageio.v3.improps(filename).shape
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)
    return shape

def check_available_gpus(gpu_id):
    """ Get the number of available GPUs
    Args:
        gpu_id (int): Number representing the requested GPU ID for segmentation
    Returns:
        n_gpus: Number of available GPUs
    """
    from torch.cuda import device_count
    n_gpus = device_count()
    if (gpu_id is not None) and (gpu_id < 0):
        logger.error("The GPU ID must be 0 or a positive integer.")
        sys.exit(3)
    elif (gpu_id is not None) and n_gpus == 0:
        logger.warning("No GPU available, using CPU.")
    elif (gpu_id is not None) and (gpu_id > n_gpus-1):
        logger.error(f"GPU ID '{str(gpu_id)}' is not available. The available GPU IDs are {str(list(range(n_gpus)))}.")
        sys.exit(3)

    return n_gpus

# Call init_ads() automatically when module is imported
# init_ads()
