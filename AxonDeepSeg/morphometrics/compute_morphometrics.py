# coding: utf-8

from pathlib import Path
from loguru import logger

# Scientific modules imports
import math
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.segmentation import watershed
import pandas as pd

# Graphs and plots imports
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from AxonDeepSeg.ads_utils import convert_path
from AxonDeepSeg import postprocessing, params
from AxonDeepSeg.visualization.colorization import colorize_instance_segmentation


def get_pixelsize(path_pixelsize_file):
    """
    :param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
    :return: the pixel size value.
    """
    
    # If string, convert to Path objects
    path_pixelsize_file = convert_path(path_pixelsize_file)

    try:
        with open(path_pixelsize_file, "r") as text_file:
            pixelsize = float(text_file.read())
    except IOError as e:
        msg = f"ERROR: Could not open file {path_pixelsize_file} from directory {Path.cwd()}."
        logger.error(msg)
        raise
    except ValueError as e:
        msg = f"ERROR: Pixel size data in file {path_pixelsize_file} is not valid - must be "\
            "a plain text file with a single numerical value (float) on the first line."
        logger.error(msg)
        raise
    else:
        return pixelsize

def get_watershed_segmentation(im_axon, im_myelin, seed_points=None):
    """
    Segments the different axonmyelin objects using a watershed algorithm. Seeds points can be passed if they are 
    already known/computed to save time.
    :param im_axon: Array: axon binary mask
    :param im_myelin: Array: myelin binary mask
    :param seed_points: (optional) Array of tuples: Seed points for the watershed algorithm. If none are provided,
    centroids of axon objects will be used. If the centroids have been computed before, it is suggested to pass them as
    seed points in order to reduce computation time.
    :return: Array containing the watershed segmentation of the axonmyelin mask
    """
    # Seed points (usually centroids) can take a while to compute, hence why there's the option to pass them directly if
    # they are computed elsewhere. If they aren't passed, we can compute them here
    if seed_points is None:
        seed_points = postprocessing.get_centroids(im_axon)

    im_axonmyelin = im_axon + im_myelin

    # Compute distance between each pixel and the background.
    distance = ndi.distance_transform_edt(im_axon)
    # Note: this distance is calculated from the im_axon,
    # not from the im_axonmyelin image, because we know that each axon
    # object is already isolated, therefore the distance metric will be
    # more useful for the watershed algorithm.

    # Create an image with axon centroids, which value corresponds to the value of the axon object
    im_centroid = np.zeros_like(im_axon, dtype='uint16')
    for i in range(len(seed_points[0])):
        # Note: The value "i" corresponds to the label number of im_axon_label
        im_centroid[seed_points[0][i], seed_points[1][i]] = i + 1

    # Watershed segmentation of axonmyelin using distance map
    return watershed(-distance, im_centroid, mask=im_axonmyelin)

def get_axon_morphometrics(
        im_axon, 
        path_folder=None, 
        im_myelin=None, 
        pixel_size=None, 
        axon_shape="circle", 
        return_index_image=False, 
        return_border_info=True,
        return_instance_seg=False,
        return_im_axonmyelin_label=False
    ):
    """
    Find each axon and compute axon-wise morphometric data, e.g., equivalent diameter, eccentricity, etc.
    If a mask of myelin is provided, also compute myelin-related metrics (myelin thickness, g-ratio, etc.).
    :param im_axon: Array: axon binary mask, output of axondeepseg
    :param path_folder: str: absolute path of folder containing pixel size file
    :param im_myelin: Array: myelin binary mask, output of axondeepseg
    :param axon_shape: str: shape of the axon, can either be either be circle or an ellipse.
                            if shape of axon = 'circle', equivalent diameter is the diameter of the axon.
                            if shape of axon = 'ellipse', ellipse minor axis is the diameter of the axon.
    :param return_index_image (optional): If set to true, an image with the index numbers at the axon centroids will be
    returned as a second return array
    :param return_border_info (optional): Flag to output if axons touch the image border along with their bounding box 

    :return: Array(dict): dictionaries containing morphometric results for each axon
    """

    if path_folder is not None:
        # If string, convert to Path objects
        path_folder = convert_path(path_folder)

        pixelsize = get_pixelsize(path_folder / 'pixel_size_in_micrometer.txt')

    if (pixel_size is not None) and (path_folder is None):
        pixelsize = pixel_size

    # Label each axon object
    connectivity = 1 if im_myelin is None else 2
    im_axon_label = measure.label(im_axon, connectivity=connectivity)
    # Measure properties for each axon object
    axon_objects = measure.regionprops(im_axon_label)

    im_shape = im_axon.shape

    # Deal with myelin mask
    if im_myelin is not None:
        # Get axon centroid as int (not float) to be used as index
        ind_centroid = ([int(props.centroid[0]) for props in axon_objects],
                        [int(props.centroid[1]) for props in axon_objects])

        im_axonmyelin_label = get_watershed_segmentation(im_axon, im_myelin, ind_centroid)
        if return_instance_seg:
            im_instance_seg = colorize_instance_segmentation(im_axonmyelin_label)
        
        # Measure properties of each axonmyelin object
        axonmyelin_objects = measure.regionprops(im_axonmyelin_label)

    # Create list of the exiting labels
    if im_myelin is not None:
        axonmyelin_labels_list = [axm.label for axm in axonmyelin_objects]

    # Declare a DataFrame that will be used to store the result of the morphometrics
    stats_dataframe = pd.DataFrame()
    
    # Loop across axon property and fill up dictionary with morphometrics of interest
    for prop_axon in axon_objects:
        # Centroid
        y0, x0 = prop_axon.centroid
        # Solidity
        solidity = prop_axon.solidity
        # Eccentricity
        eccentricity = prop_axon.eccentricity
        if axon_shape == "circle":
            # Axon equivalent diameter in micrometers
            axon_diam = prop_axon.equivalent_diameter * pixelsize
        else: 
            # Axon diameter using ellipse minor axis in micrometers
            axon_diam = prop_axon.minor_axis_length * pixelsize
        # Axon area in µm^2
        axon_area = prop_axon.area * (pixelsize ** 2)
        # Axon perimeter (inner perimeter of myelin) in micrometers
        axon_perimeter = prop_axon.perimeter * pixelsize
        # Axon orientation angle
        orientation = prop_axon.orientation
        # Add metrics to list of dictionaries
        stats = {'x0': x0,
                 'y0': y0,
                 'axon_diam': axon_diam,
                 'axon_area': axon_area,
                 'axon_perimeter': axon_perimeter,
                 'solidity': solidity,
                 'eccentricity': eccentricity,
                 'orientation': orientation
                 }

        # Deal with myelin
        if im_myelin is not None:
            # Declare the statistics to add for the myelin and add them to the stats dictionary
            myelin_stats = {
                'gratio': np.nan,
                'myelin_thickness': np.nan,
                'myelin_area': np.nan,
                'axonmyelin_area': np.nan,
                'axonmyelin_perimeter': np.nan
            }
            stats.update(myelin_stats)

            # Find label of axonmyelin corresponding to axon centroid
            label_axonmyelin = im_axonmyelin_label[int(y0), int(x0)]

            if label_axonmyelin:
                idx = axonmyelin_labels_list.index(label_axonmyelin)
                prop_axonmyelin = axonmyelin_objects[idx]

                _res1 = evaluate_myelin_thickness_in_px(prop_axon, prop_axonmyelin, axon_shape)
                myelin_thickness = pixelsize * _res1

                _res2 = evaluate_myelin_area_in_px(prop_axon, prop_axonmyelin)
                myelin_area = (pixelsize ** 2) * _res2

                axonmyelin_area = (pixelsize ** 2) * prop_axonmyelin.area
                # Perimeter of axonmyelin instance (outer perimeter of myelin) in micrometers
                axonmyelin_perimeter = prop_axonmyelin.perimeter * pixelsize

                try:
                    stats['gratio'] = (axon_diam / 2) / (axon_diam / 2 + myelin_thickness)
                    stats['myelin_thickness'] = myelin_thickness
                    stats['myelin_area'] = myelin_area
                    stats['axonmyelin_area'] = axonmyelin_area
                    stats['axonmyelin_perimeter'] = axonmyelin_perimeter
                except ZeroDivisionError:
                    logger.warning(f"ZeroDivisionError caught on invalid object #{idx}.")
                    stats['gratio'] = np.nan
                    stats['myelin_thickness'] = np.nan
                    stats['myelin_area'] = np.nan
                    stats['axonmyelin_area'] = np.nan
                    stats['axonmyelin_perimeter'] = np.nan
                
                if return_border_info:
                    # check if bounding box touches a border (partial axonmyelin object)
                    bbox = prop_axonmyelin.bbox
                    touching = 0 in bbox[:2] or bbox[2] == im_shape[0] or bbox[3] == im_shape[1]
                    border_info_stats = {
                        'image_border_touching': touching,
                        'bbox_min_y': bbox[0],
                        'bbox_min_x': bbox[1],
                        'bbox_max_y': bbox[2],
                        'bbox_max_x': bbox[3]
                    }
                    stats.update(border_info_stats)

            else:
                logger.warning(f"WARNING: Myelin object not found for axon centroid [y:{y0}, x:{x0}]")

        # Add the stats to the dataframe
        stats_dataframe = pd.concat([stats_dataframe, pd.DataFrame(stats, index=[0])], ignore_index=True)

    if stats_dataframe.empty:
        stats_dataframe = pd.DataFrame(columns=['x0','y0','axon_diam','axon_area','axon_perimeter', 'gratio', 'myelin_thickness', 'myelin_area', 'solidity','eccentricity', 'orientation']) 
        empty_df = True
    else: 
        empty_df = False

    if (not return_index_image) and (not return_instance_seg):
        return stats_dataframe
    else:
        output = (stats_dataframe,)
        
    if return_index_image is True: 
        if empty_df:
            index_image_array = np.zeros_like(im_axon, dtype='uint8')
            output = (*output, index_image_array)
        else:
            # Extract the information required to generate the index image
            x0_array = stats_dataframe["x0"].to_numpy()
            y0_array = stats_dataframe["y0"].to_numpy()
            diam_array = stats_dataframe["axon_diam"].to_numpy()
            # Create the axon coordinate array, then generate the image
            mean_diameter_in_pixel = np.average(diam_array) / pixelsize
            axon_indexes = np.arange(stats_dataframe.shape[0])
            index_image_array = postprocessing.generate_axon_numbers_image(axon_indexes, x0_array, y0_array,
                                                                    tuple(reversed(im_axon.shape)),
                                                                    mean_diameter_in_pixel)
            output = (*output, index_image_array)
    
    if return_instance_seg:
        output = (*output, im_instance_seg)
    
    if return_im_axonmyelin_label:
        output = (*output, im_axonmyelin_label)

    return output

def evaluate_myelin_thickness_in_px(axon_object, axonmyelin_object, axon_shape):
    """
    Returns the  thickness of a myelin ring around an axon of a
    given equivalent diameter (see note [1] below) or minimum axis diameter of an ellipse. The result is in pixels.
    :param axon_object (skimage.measure._regionprops): object returned after
        measuring a axon labeled region
    :param axonmyelin_object (skimage.measure._regionprops): object returned after
        measuring a axon with myelin labeled region
    :param axon_shape: str: shape of the axon, can either be either be circle or an ellipse.
                            if shape of axon = 'circle', equivalent diameter is the diameter of the axon.
                            if shape of axon = 'ellipse', ellipse minor axis is the diameter of the axon.
    

    [1] According to https://scikit-image.org/docs/dev/api/skimage.measure.html?highlight=region%20properties#regionprops,
    the equivalent diameter is the diameter of a circle with the same area as
    the region.
    """
    attribute = "equivalent_diameter" if axon_shape == "circle" else "minor_axis_length" 
    warn_if_measures_are_unexpected(
        axon_object,
        axonmyelin_object,
        attribute
        )
    if axon_shape == "circle": 
        axon_diam = axon_object.equivalent_diameter
        axonmyelin_diam = axonmyelin_object.equivalent_diameter
    else: 
        axon_diam = axon_object.minor_axis_length
        axonmyelin_diam = axonmyelin_object.minor_axis_length
    return (axonmyelin_diam - axon_diam)/2

def evaluate_myelin_area_in_px(axon_object, axonmyelin_object):
    """
    Returns the myenlinated axon area minus the axon area.
    :param axon_object (skimage.measure._regionprops): object returned after
    measuring an axon labeled  region
    :param axonmyelin_object (skimage.measure._regionprops): object returned after
    measuring a axon with myelin labeled region
    """
    warn_if_measures_are_unexpected(
        axon_object,
        axonmyelin_object,
        "area"
        )
    return axonmyelin_object.area - axon_object.area

def warn_if_measures_are_unexpected(axon_object, axonmyelin_object, attribute):
    """
    Calls the `_check_measures_are_relatively_valid` function and if return
    value is False, print a warning.
    """
    checked = _check_measures_are_relatively_valid(axon_object, axonmyelin_object, attribute)
    if checked is False:
        x_a, y_a = axon_object.centroid
        msg = (
            f"WARNING: Axon {axon_object.label} at [y:{y_a}, x:{x_a}] and corresponding "
            f"myelinated axon {axonmyelin_object.label} have unexpected measure values "
            f"for {attribute} attribute."
        )
        logger.warning(msg)

def _check_measures_are_relatively_valid(axon_object, axonmyelin_object, attribute):
    """
    Checks if the attribute is positive and if the myelinated axon has a greater value
    """
    val_axon = getattr(axon_object, attribute)
    val_axonmyelin = getattr(axonmyelin_object, attribute)
    if val_axon > 0 and val_axonmyelin > 0 and val_axonmyelin > val_axon:
        return True
    else:
        return False

def rearrange_column_names_for_saving(stats_dataframe):
    """
    This function rearranges the columns in the stats_dataframe to match the order and display names specified in
    params.py.
    :param stats_dataframe: the dataframe with the morphometrics data
    :return: the dataframe with the ordered columns and units in the column names
    """
    # Reorder the columns
    order_of_columns = []
    columns_rename_dict = {}
    for column in params.column_names_ordered:
        if column.key_name in stats_dataframe.columns:
            order_of_columns.append(column.key_name)
            if column.display_name is not None:
                columns_rename_dict[column.key_name] = column.display_name
    for column in stats_dataframe.columns: # Add the remaining columns for which no unit or order was specified
        if column not in order_of_columns:
            order_of_columns.append(column)
    stats_dataframe = stats_dataframe[order_of_columns]

    # Add units or other details to the column names
    stats_dataframe = stats_dataframe.rename(columns=columns_rename_dict)
    return stats_dataframe

def rename_column_names_after_loading(loaded_dataframe):
    """
    This function removes the units in the loaded_dataframe column names so that the dataframe can be used more easily
    internally.
    :param loaded_dataframe: The dataframe that was loaded, containing the morphometrics data
    :return: the dataframe without the units in the column names
    """
    columns_rename_dict = {}
    for column in params.column_names_ordered:
        if (column.display_name is not None) and (column.display_name in loaded_dataframe.columns):
            columns_rename_dict[column.display_name] = column.key_name
    loaded_dataframe = loaded_dataframe.rename(columns=columns_rename_dict)
    return loaded_dataframe

def save_axon_morphometrics(morphometrics_file, stats_dataframe):
    """
    :param morphometrics_file: absolute path of file that will be saved (with the extension)
    :param stats_dataframe: dataframe containing the morphometrics
    """
    
    # If string, convert to Path objects
    morphometrics_file = convert_path(morphometrics_file)
    if morphometrics_file.suffix == "":
        raise ValueError("Invalid file name. Please include its name and extension")

    stats_dataframe = rearrange_column_names_for_saving(stats_dataframe)
    if morphometrics_file.suffix.lower() == ".csv":  # Export to csv
        stats_dataframe.to_csv(morphometrics_file, na_rep='NaN')
    elif morphometrics_file.suffix.lower() == ".xlsx":  # Export to excel
        stats_dataframe.to_excel(morphometrics_file, na_rep='NaN')
    else:  # Export to pickle
        stats_dataframe.to_pickle(morphometrics_file)

def load_axon_morphometrics(morphometrics_file):
    """
    :param morphometrics_file: absolute path of file containing the morphometrics (must be .csv, .xlsx or pickle format)
    :return: stats_dataframe: dataframe containing the morphometrics
    """
    
    # If string, convert to Path objects
    morphometrics_file = convert_path(morphometrics_file)

    if morphometrics_file.suffix == "":
        raise ValueError("File not specified. Please provide the full path of the file, including its extension")

    try:
        #Use the appropriate loader depending on the extension
        if morphometrics_file.suffix.lower() == ".csv":
            stats_dataframe = pd.read_csv(morphometrics_file, na_values='NaN')
        elif morphometrics_file.suffix.lower() == ".xlsx":
            stats_dataframe = pd.read_excel(morphometrics_file, na_values='NaN')
        else:
            stats_dataframe = pd.read_pickle(morphometrics_file)
    except IOError as e:
        logger.error(f"Error: Could not load file {str(morphometrics_file)}")
        raise

    stats_dataframe = rename_column_names_after_loading(stats_dataframe)
    # with csv and excel files, they often will have an "unnamed" column because of the indexes saved with the dataframe
    # we remove it here
    for column in stats_dataframe.columns:
        if "unnamed" in column.lower():
            stats_dataframe = stats_dataframe.drop(columns=column)

    return stats_dataframe

def draw_axon_diameter(img, path_prediction, pred_axon, pred_myelin, axon_shape="circle"):
    """
    :param img: sample grayscale image (png)
    :param path_prediction: full path to the segmented file (*_seg-axonmyelin.png)
        from axondeepseg segmentation output
    :param pred_axon: axon mask from axondeepseg segmentation output
    :param pred_myelin: myelin mask from axondeepseg segmentation output
    :param axon_shape: str: shape of the axon, can either be either be circle or an ellipse.
                            if shape of axon = 'circle', equivalent diameter is the diameter of the axon.
                            if shape of axon = 'ellipse', ellipse minor axis is the diameter of the axon.
    :return: matplotlib.figure.Figure
    """
    
    # If string, convert to Path objects
    path_prediction = convert_path(path_prediction)

    path_folder = path_prediction.parent

    stats_dataframe = get_axon_morphometrics(pred_axon, path_folder, axon_shape=axon_shape)
    axon_diam_array = stats_dataframe["axon_diam"].to_numpy()

    labels = measure.label(pred_axon)
    axon_diam_display = np.zeros((np.shape(labels)[0], np.shape(labels)[1]))

    for pix_x in np.arange(np.shape(labels)[0]):
        for pix_y in np.arange(np.shape(labels)[1]):
            if labels[pix_x, pix_y] != 0:
                axon_diam_display[pix_x, pix_y] = axon_diam_array[
                    labels[pix_x, pix_y] - 1
                ]

    # Axon overlay on original image + myelin display (same color for every
    # myelin sheath)
    fig = Figure(figsize=(12, 9))
    FigureCanvas(fig)
    ax = fig.subplots()
    ax.imshow(img, cmap="gray", alpha=0.8)
    ax.imshow(pred_myelin, cmap="gray", alpha=0.3)
    im = ax.imshow(axon_diam_display, cmap="hot", alpha=0.5)
    fig.colorbar(im, fraction=0.03, pad=0.02)
    ax.set_title(
        "Axon overlay (colorcoded with axon diameter in um) and myelin display",
        fontsize=12,
    )
    return fig

def save_map_of_axon_diameters(path_folder, axon_diameter_figure):
    """
    :param path_folder: absolute path of the sample and the segmentation folder
    :param axon_diameter_figure: figure create with draw_axon_diameter
    :return: None
    """
    
    # If string, convert to Path objects
    path_folder = convert_path(path_folder)
    
    file_path = path_folder / "AxonDeepSeg_map-axondiameter.png"
    axon_diameter_figure.savefig(file_path)

def get_aggregate_morphometrics(pred_axon, pred_myelin, path_folder, axon_shape="circle"):
    """
    :param pred_axon: axon mask from axondeepseg segmentation output
    :param pred_myelin: myelin mask from axondeepseg segmentation output
    :param path_folder: absolute path of folder containing pixel size file
    :param axon_shape: str: shape of the axon, can either be either be circle or an ellipse.
                            if shape of axon = 'circle', equivalent diameter is the diameter of the axon.
                            if shape of axon = 'ellipse', ellipse minor axis is the diameter of the axon.
    :return: aggregate_metrics: dictionary containing values of aggregate metrics
    """

    # If string, convert to Path objects
    path_folder = convert_path(path_folder)

    # Compute AVF (axon volume fraction) = area occupied by axons in sample
    avf = np.count_nonzero(pred_axon) / float((pred_axon.size))
    # Compute MVF (myelin volume fraction) = area occupied by myelin sheaths in sample
    mvf = np.count_nonzero(pred_myelin) / float((pred_myelin.size))

    # Estimate aggregate g-ratio = sqrt(1/(1+MVF/AVF))
    gratio = math.sqrt(1 / (1 + (float(mvf) / float(avf))))

    # Get individual axons metrics and compute mean axon diameter
    stats_dataframe = get_axon_morphometrics(pred_axon, path_folder, axon_shape=axon_shape)
    axon_diam_list = stats_dataframe["axon_diam"].to_list()
    mean_axon_diam = np.mean(axon_diam_list)

    # Estimate mean myelin diameter (axon+myelin diameter) by using
    # aggregate g-ratio = mean_axon_diam/mean_myelin_diam

    mean_myelin_diam = mean_axon_diam / gratio

    # Estimate mean myelin thickness = mean_myelin_radius - mean_axon_radius
    mean_myelin_thickness = (float(mean_myelin_diam) / 2) - (float(mean_axon_diam) / 2)

    # Compute axon density (number of axons per mm2)
    px_size_um = get_pixelsize(path_folder / 'pixel_size_in_micrometer.txt')
    img_area_mm2 = float(pred_axon.size) * px_size_um * px_size_um / (float(1000000))
    axon_density_mm2 = float(len(axon_diam_list)) / float(img_area_mm2)

    # Create disctionary to store aggregate metrics
    aggregate_metrics = {'avf': avf, 'mvf': mvf, 'gratio_aggr': gratio, 'mean_axon_diam': mean_axon_diam,
                         'mean_myelin_diam': mean_myelin_diam, 'mean_myelin_thickness': mean_myelin_thickness,
                         'axon_density_mm2': axon_density_mm2}
    return aggregate_metrics

def write_aggregate_morphometrics(path_folder, aggregate_metrics):
    """
    :param path_folder: absolute path of folder containing sample + segmentation
    :param aggregate_metrics: dictionary containing values of aggregate metrics
    :return: nothing
    """
    
    # If string, convert to Path objects
    path_folder = convert_path(path_folder)
    
    try:
        with open(path_folder / 'aggregate_morphometrics.txt', 'w') as text_file:
            text_file.write('aggregate_metrics: ' + repr(aggregate_metrics) + '\n')
    except IOError as e:
        msg = f"Error: Could not save file \"aggregate_morphometrics.txt\" in directory \"{path_folder}\"."
        logger.error(msg)
        raise
