# coding: utf-8

from scipy import ndimage as ndi
import os
import math
from sys import platform as _platform
if _platform == "darwin": # Mac OSX
    import matplotlib as mpl
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import measure, morphology, feature
from AxonDeepSeg.testing.segmentation_scoring import *


def get_pixelsize(path_pixelsize_file):
    """
    :param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
    :return: the pixel size value.
    """
    try:
        with open(path_pixelsize_file, "r") as text_file:
            pixelsize = float(text_file.read())
    except IOError as e:
        print(("\nError: Could not open file \"{0}\" from "
               "directory \"{1}\".\n".format(path_pixelsize_file, os.getcwd())))
        raise
    except ValueError as e:
        print(("\nError: Pixel size data in file \"{0}\" is not valid – must "
               "be a plain text file with a single a numerical value (float) "
               " on the fist line.".format(path_pixelsize_file)))
        raise
    else:
        return pixelsize


def get_axon_morphometrics(im_axon, path_folder, im_myelin=None):
    """
    Find each axon and compute axon-wise morphometric data, e.g., equivalent diameter, eccentricity, etc.
    If a mask of myelin is provided, also compute myelin-related metrics (myelin thickness, g-ratio, etc.).
    :param im_axon: Array: axon binary mask, output of axondeepseg
    :param path_folder: str: absolute path of folder containing pixel size file
    :param im_myelin: Array: myelin binary mask, output of axondeepseg
    :return: Array(dict): dictionaries containing morphometric results for each axon
    """
    # TODO: externalize reading of pixel_size_in_micrometer.txt and input float
    pixelsize = get_pixelsize(os.path.join(path_folder, 'pixel_size_in_micrometer.txt'))
    stats_array = np.empty(0)
    # Label each axon object
    im_axon_label = measure.label(im_axon)
    # Measure properties for each axon object
    axon_objects = measure.regionprops(im_axon_label)
    # Deal with myelin mask
    if im_myelin is not None:
        # sum axon and myelin masks
        im_axonmyelin = im_axon + im_myelin
        # Compute distance between each pixel and the background. Note: this distance is calculated from the im_axon,
        # note from the im_axonmyelin image, because we know that each axon object is already isolated, therefore the
        # distance metric will be more useful for the watershed algorithm below.
        distance = ndi.distance_transform_edt(im_axon)
        # local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((31, 31)), labels=axonmyelin)

        # Get axon centroid as int (not float) to be used as index
        ind_centroid = ([int(props.centroid[0]) for props in axon_objects],
                        [int(props.centroid[1]) for props in axon_objects])

        # Create an image with axon centroids, which value corresponds to the value of the axon object
        im_centroid = np.zeros_like(im_axon)
        for i in range(len(ind_centroid[0])):
            # Note: The value "i" corresponds to the label number of im_axon_label
            im_centroid[ind_centroid[0][i], ind_centroid[1][i]] = i

        # markers = ndi.label(local_maxi)[0]
        # Watershed segmentation of axonmyelin using distance map
        im_axonmyelin_label = morphology.watershed(-distance, im_centroid, mask=im_axonmyelin)
        # Measure properties of each axonmyelin object
        axonmyelin_objects = measure.regionprops(im_axonmyelin_label)

    # DEBUG
    # from matplotlib import colors
    # from matplotlib.pylab import *
    # random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
    # matshow(im_axon_label, fignum=1, cmap=random_cmap), show()
    # import datetime
    # savefig('fig_' + datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S%f') + '.png', format='png',
    #         transparent=True, dpi=100)

    # Loop across axon property and fill up dictionary with morphometrics of interest
    for prop_axon in axon_objects:
        # Centroid
        y0, x0 = prop_axon.centroid
        # Solidity
        solidity = prop_axon.solidity
        # Eccentricity
        eccentricity = prop_axon.eccentricity
        # Axon equivalent diameter in micrometers
        axon_diam = prop_axon.equivalent_diameter * pixelsize
        # Axon orientation angle
        orientation = prop_axon.orientation
        # Add metrics to list of dictionaries
        stats = {'y0': y0,
                 'x0': x0,
                 'axon_diam': axon_diam,
                 'solidity': solidity,
                 'eccentricity': eccentricity,
                 'orientation': orientation}
        # Deal with myelin
        if im_myelin is not None:
            # Find label of axonmyelin corresponding to axon centroid
            label_axonmyelin = im_axonmyelin_label[int(y0), int(x0)]
            # TODO: use logger
            # print(label_axonmyelin)
            # print('x, y = {}, {}'.format(x0, y0))
            if label_axonmyelin:
                # Get corresponding index from axonmyelin_objects list
                ind_axonmyelin = \
                    [axonmyelin_object.label for axonmyelin_object in axonmyelin_objects].index(label_axonmyelin)
                myelin_diam = axonmyelin_objects[ind_axonmyelin].equivalent_diameter * pixelsize
                stats['myelin_diam'] = myelin_diam
                stats['gratio'] = axon_diam / myelin_diam
            else:
                # TODO: use logger
                print('WARNING: Myelin object not found for axon centroid [{},{}]'.format(y0, x0))

        stats_array = np.append(stats_array, [stats], axis=0)

    return stats_array


def save_axon_morphometrics(path_folder, stats_array):
    """
    :param path_folder: absolute path of folder containing the sample + the segmentation output
    :param stats_array: list of dictionaries containing axon morphometrics
    :return:
    """
    try:
        np.save(os.path.join(path_folder, 'axonlist.npy'), stats_array)
    except IOError as e:
        print(("\nError: Could not save file \"{0}\" in "
               "directory \"{1}\".\n".format('axonlist.npy', path_folder)))
        raise


def load_axon_morphometrics(path_folder):
    """
    :param path_folder: absolute path of folder containing the sample + the segmentation output
    :return: stats_array: list of dictionaries containing axon morphometrics
    """
    try:
        stats_array = np.load(os.path.join(path_folder, 'axonlist.npy'))
    except IOError as e:
        print(("\nError: Could not load file \"{0}\" in "
               "directory \"{1}\".\n".format('axonlist.npy', path_folder)))
        raise
    else:
        return stats_array


def display_axon_diameter(img, path_prediction, pred_axon, pred_myelin):
    """
    :param img: sample grayscale image (png)
    :param path_prediction: full path to the segmented file (*_seg-axonmyelin.png) from axondeepseg segmentation output
    :param pred_axon: axon mask from axondeepseg segmentation output
    :param pred_myelin: myelin mask from axondeepseg segmentation output
    :return: nothing
    """
    path_folder, file_name = os.path.split(path_prediction)
    tmp_path = path_prediction.split('_seg-axonmyelin')

    stats_array = get_axon_morphometrics(pred_axon, path_folder)
    axon_diam_list = [d['axon_diam'] for d in stats_array]
    axon_diam_array = np.asarray(axon_diam_list)
    axon_iter = np.arange(np.size(axon_diam_array))

    labels = measure.label(pred_axon)
    axon_diam_display = a = np.zeros((np.shape(labels)[0], np.shape(labels)[1]))

    for pix_x in np.arange(np.shape(labels)[0]):
        for pix_y in np.arange(np.shape(labels)[1]):
            if labels[pix_x, pix_y] != 0:
                axon_diam_display[pix_x, pix_y] = axon_diam_array[labels[pix_x, pix_y] - 1]

    # Axon overlay on original image + myelin display (same color for every myelin sheath)
    plt.figure(figsize=(12, 9))
    plt.imshow(img, cmap='gray', alpha=0.8)
    plt.imshow(pred_myelin, cmap='gray', alpha=0.3)
    im = plt.imshow(axon_diam_display, cmap='hot', alpha=0.5)
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.title('Axon overlay (colorcoded with axon diameter in um) and myelin display', fontsize=12)
    plt.savefig(tmp_path[0] + '_map-axondiameter.png')


def get_aggregate_morphometrics(pred_axon, pred_myelin, path_folder):
    """
    :param pred_axon: axon mask from axondeepseg segmentation output
    :param pred_myelin: myelin mask from axondeepseg segmentation output
    :param path_folder: absolute path of folder containing pixel size file
    :return: aggregate_metrics: dictionary containing values of aggregate metrics
    """

    # Compute AVF (axon volume fraction) = area occupied by axons in sample
    avf = np.count_nonzero(pred_axon) / float((pred_axon.size))
    # Compute MVF (myelin volume fraction) = area occupied by myelin sheaths in sample
    mvf = np.count_nonzero(pred_myelin) / float((pred_myelin.size))

    # Estimate aggregate g-ratio = sqrt(1/(1+MVF/AVF))
    gratio = math.sqrt(1 / (1 + (float(mvf) / float(avf))))

    # Get individual axons metrics and compute mean axon diameter
    stats_array = get_axon_morphometrics(pred_axon, path_folder)
    axon_diam_list = [d['axon_diam'] for d in stats_array]
    mean_axon_diam = np.mean(axon_diam_list)

    # Estimate mean myelin diameter (axon+myelin diameter) by using aggregate g-ratio = mean_axon_diam/mean_myelin_diam
    mean_myelin_diam = mean_axon_diam / gratio

    # Estimate mean myelin thickness = mean_myelin_radius - mean_axon_radius
    mean_myelin_thickness = (float(mean_myelin_diam) / 2) - (float(mean_axon_diam) / 2)

    # Compute axon density (number of axons per mm2)
    y, x = pred_axon.shape
    img_area_mm2 = float(pred_axon.size) * get_pixelsize(
        os.path.join(path_folder, 'pixel_size_in_micrometer.txt')) * get_pixelsize(
        os.path.join(path_folder, 'pixel_size_in_micrometer.txt')) / (float(1000000))
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
    try:
        with open(os.path.join(path_folder, 'aggregate_morphometrics.txt'), 'w') as text_file:
            text_file.write('aggregate_metrics: ' + repr(aggregate_metrics) + '\n')
    except IOError as e:
        print(("\nError: Could not save file \"{0}\" in "
               "directory \"{1}\".\n".format('aggregate_morphometrics.txt', path_folder)))
        raise
