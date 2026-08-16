"""
AxonDeepSeg utilities module.
"""

import json
import os
import sys
from pathlib import Path
import tempfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from tqdm import tqdm
import imageio
import numpy as np
from PIL import Image
from loguru import logger

from AxonDeepSeg.params import valid_extensions
import re

# Raised pixel limit for large scientific images.
# PIL's default is ~178M pixels; 1B pixels accommodates most microscopy use cases.
_LARGE_IMAGE_PIXEL_LIMIT = 1_000_000_000

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
            header = response.headers["Content-Disposition"]
            
            # Extract filename manually using regex
            match = re.search(r'filename="?(?P<filename>[^";]+)"?', header)
            if match:
                zip_filename = match.group("filename")
            else:
                print("Unexpected: Unable to extract filename from Content-Disposition")
        else:
            print("Unexpected: link doesn't provide a filename")

        # Save to temporary directory
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

def imread(filename, use_16bit=False, allow_large_images=False):
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

    # Check if the image exceeds PIL's decompression bomb pixel limit.
    shape = get_imshape(filename, allow_large_images=allow_large_images)
    n_pixels = shape[0] * shape[1]
    if Image.MAX_IMAGE_PIXELS is not None and n_pixels > Image.MAX_IMAGE_PIXELS:
        if not allow_large_images:
            raise IOError(
                f"'{filename.name}' has {n_pixels:,} pixels, which exceeds PIL's decompression "
                f"bomb limit of {Image.MAX_IMAGE_PIXELS:,} pixels. Re-run with "
                f"--allow-large-images to process it."
            )
        logger.warning(
            f"'{filename.name}' has {n_pixels:,} pixels, which exceeds PIL's default "
            f"decompression bomb limit. Processing with a raised limit of "
            f"{_LARGE_IMAGE_PIXEL_LIMIT:,} pixels."
        )

    old_limit = Image.MAX_IMAGE_PIXELS
    if allow_large_images:
        Image.MAX_IMAGE_PIXELS = _LARGE_IMAGE_PIXEL_LIMIT
    try:
        props = imageio.v3.improps(filename) # new in v3 - standardized metadata
        _img = imageio.v3.imread(filename)

        if _img.dtype==np.float32 or _img.dtype==np.float64:
            if len(_img.shape) > 2:
                raise IOError(f"Multichannel 32bit and 64bit float images are not supported. Please convert the image to 8bit or 16bit, or make a single channel image.")

        # brute force fall back to support backward compatibility
        if '.tif' in file_ext:
            img = np.expand_dims(
                imageio.v3.imread(filename, plugin='TIFF-PIL'),
                axis=-1).astype(np.uint8)

            if len(img.shape) > 3:
                img = np.expand_dims(
                    imageio.v3.imread(filename, plugin='TIFF-PIL', as_gray=True),
                    axis=-1).astype(np.uint8)

        # c.f for more details: https://github.com/ivadomed/ivadomed/pull/1297#discussion_r1267563980 and
        # https://github.com/ivadomed/ivadomed/pull/1297#discussion_r1267563980

        # TIFF is a "wild" format. A few assumptions greatly simplify the code below:
        # 1. the image is interleaved/channel-last (not planar)
        # 2. the colorspace is one of: binary, gray, RGB, RGBA (not aliasing ones like YUV or CMYK)
        # 3. the image is flat (not a volume or time-series)
        # Note: All of these are trivially true for JPEG and PNG due to limitations of these formats.

        # make grayscale (treats binary as 1-bit grayscale)
        colorspace_idx = 2
        if _img.ndim <= colorspace_idx:  # binary or gray
            pass  # nothing to do
        elif _img.shape[colorspace_idx] == 2:  # gray with alpha channel
            _img = _img[:, :, 0]
        elif _img.shape[colorspace_idx] == 3:  # RGB
            _img = np.sum(_img * (.299, .587, .114), axis=-1)
        else:  # RGBA
            # discards alpha
            _img = np.sum(_img * (.299, .587, .114, 0), axis=-1)
        if len(_img.shape) < 3:
            _img = np.expand_dims(_img, axis=-1)

        bitdepth = 16 if use_16bit else 8
        img = imageio.core.image_as_uint(_img, bitdepth=bitdepth)

        # Remove singleton dimension
        if img.ndim == 3 and img.shape[-1] == 1:
           img = img[:, :, 0]
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit

    return img

def imwrite(filename, img, format='png', use_16bit=False):
    """ Write image.
    """
    # check datatype:
    conversion_list = ['float64', 'float32', 'float16']
    if not use_16bit:
        conversion_list.append('uint16')

    if img.dtype in conversion_list:
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

    # If models folder doesn't exist or it's empty, return None
    if not os.path.exists(models_path) or not os.listdir(models_path):
        return None

    models_list = next(os.walk(models_path))[1]
    if "__pycache__" in models_list:
        models_list.remove("__pycache__")
    return models_list

def get_model_output_classes(path_model: str):
    """
    This method returns the output classes of a model located under AxonDeepSeg/models
    :param path_model: path to the model folder
    :return: list containing the output classes of the model
    :rtype: list of strings
    """
    dataset_json_path = Path(path_model) / "dataset.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found in {path_model}. Please make sure the model is valid.")

    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)

    output_structure = dataset_json['labels']
    output_classes = list(output_structure.keys())
    output_classes.remove('background')
    return output_classes

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

def get_imshape(filename: str, allow_large_images: bool = False):
    """Get the shape of an image (HWC format) without reading its data.
    """
    old_limit = Image.MAX_IMAGE_PIXELS
    if allow_large_images:
        Image.MAX_IMAGE_PIXELS = _LARGE_IMAGE_PIXEL_LIMIT
    try:
        shape = imageio.v3.improps(filename).shape
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
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
