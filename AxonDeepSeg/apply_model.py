from pathlib import Path
import json

# AxonDeepSeg imports
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.visualization.merge_masks import merge_masks
from config import axon_suffix, myelin_suffix, axonmyelin_suffix

from ivadomed import inference as imed_inference

def axon_segmentation(
                    path_acquisitions_folders, 
                    acquisitions_filenames,
                    path_model_folder,
                    overlap_value=[48,48],
                    acquired_resolution=None,
                    verbosity_level = 0
                    ):
    '''
    Segment images using IVADOMED.
    :param path_acquisitions_folders: the directory containing the images to segment.
    :param acquisitions_filenames: filenames of the images to segment.
    :param path_model_folder: path to the folder of the IVADOMED-trained model.
    :param overlap_value: the number of pixels to be used for overlap when doing prediction. Higher value means less
    border effects but more time to perform the segmentation.
    :param acquired_resolution: isotropic pixel size of the acquired images.
    :param verbosity_level: Level of verbosity. The higher, the more information is given about the segmentation
    process.
    :return: Nothing.
    '''

    # If we did not receive any resolution we read the pixel size in micrometer from each pixel.
    if acquired_resolution == None:
        if (path_acquisitions_folders / 'pixel_size_in_micrometer.txt').exists():
            resolutions_file = open(path_acquisitions_folders / 'pixel_size_in_micrometer.txt', 'r')
            str_resolution = float(resolutions_file.read())
            acquired_resolution = float(str_resolution)
        else:
            exception_msg = "ERROR: No pixel size is provided, and there is no pixel_size_in_micrometer.txt file in image folder. " \
                            "Please provide a pixel size (using argument acquired_resolution), or add a pixel_size_in_micrometer.txt file " \
                            "containing the pixel size value."
            raise Exception(exception_msg)

    path_model=path_model_folder
    input_filenames = acquisitions_filenames
    options = {"pixel_size": [acquired_resolution, acquired_resolution], "pixel_size_units": "um", "overlap_2D": overlap_value, "binarize_maxpooling": True}

    # IVADOMED automated segmentation
    try:
        nii_lst, _ = imed_inference.segment_volume(str(path_model), input_filenames, options=options)
    except RuntimeError as err:
        # check minimum patch size requirement
        px_size = options["pixel_size"][0]
        model_json = list(Path(path_model).glob('*.json'))
        with open(str(model_json[0]), 'r') as param_file:
            params = json.load(param_file)
            length_2D = params["default_model"]["length_2D"][0]
            model_res = params["transformation"]["Resample"]["wspace"]*1000
        for file in input_filenames:
            img = ads.imread(file)
            resampled = [size * px_size / model_res for size in img.shape]
            smallest_dim = sorted(resampled)[0]
            if smallest_dim < length_2D:
                min_zoom = length_2D / smallest_dim
                msg = (f"The image size must be at least {length_2D}x{length_2D} after resampling to a resolution " 
                    f"of {model_res} um/pixels to create standard sized patches. \nOne of the dimensions of the "
                    f"image has a size of {int(smallest_dim)} after resampling to that resolution. \n"
                    f"Please use a zoom factor greater or equal to {min_zoom:.2f} using the `-z` option to successfully apply this model.")
                raise RuntimeError(msg) from err
        raise
    
    target_lst = [str(axon_suffix), str(myelin_suffix)]

    imed_inference.pred_to_png(nii_lst, target_lst, str(Path(input_filenames[0]).parent / Path(input_filenames[0]).stem))
    if verbosity_level >= 1:
        print(Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axonmyelin_suffix)))
    
    merge_masks(
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axon_suffix)),
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(myelin_suffix)), 
        Path(path_acquisitions_folders) / (Path(input_filenames[0]).stem + str(axonmyelin_suffix))
        )
