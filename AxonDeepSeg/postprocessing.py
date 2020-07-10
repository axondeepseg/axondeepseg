""""
Tools for the FSLeyes plugin.
"""
from skimage import measure, morphology, feature
from PIL import Image, ImageDraw, ImageOps
import numpy as np

def get_centroids(mask):
    """
    This function is used to find the centroids of objects found in the mask
    :param mask: the binary mask.
    :type mask: ndarray
    :return: a list containing the coordinates of the centroid of every object
    :rtype: list of int
    """
    # Label each object
    object_label = measure.label(mask)
    # Find the centroids of the objects
    objects = measure.regionprops(object_label)
    ind_centroid = (
        [int(props.centroid[0]) for props in objects],
        [int(props.centroid[1]) for props in objects],
    )
    return ind_centroid

def floodfill_axons(axon_array, myelin_array):
    """
    This function does a floodfill operation on the myelin_array. The seed points are the centroids of the axon objects
    in the axon_array. The goal is to fill the center of the myelin objects with the axon mask.
    Note: The myelin objects need to be closed in order to prevent the axon_mask from flooding the entire image.
    :param axon_array: the binary array corresponding to the axon mask
    :param myelin_array: the binary array corresponding to the myelin mask
    :return: the binary axon array corresponding to the axon mask after the floodfill
    """
    # Get the centroid indexes
    centroid_index_map = get_centroids(axon_array)

    # Create an image with the myelinMask and floodfill at the coordinates of the centroids
    # Note: The floodfill algorithm only works on RGB images. Thus, the mask must be colorized before applying
    # the floodfill. Then, the array corresponding to the floodfilled color can be extracted.
    myelin_image = Image.fromarray(myelin_array * 255)
    myelin_image = ImageOps.colorize(
        myelin_image, (0, 0, 0, 255), (255, 255, 255, 255)
    )
    for i in range(len(centroid_index_map[0])):
        ImageDraw.floodfill(
            myelin_image,
            xy=(centroid_index_map[1][i], centroid_index_map[0][i]),
            value=(127, 127, 127, 255),
        )

    # Extract the axon_mask overlay
    axon_extracted_array = np.array(myelin_image.convert("LA"))
    axon_extracted_array = axon_extracted_array[:, :, 0]
    axon_extracted_array = np.equal(
        axon_extracted_array, 127 * np.ones_like(axon_extracted_array)
    )
    axon_extracted_array = axon_extracted_array.astype(np.uint8)
    return axon_extracted_array

def remove_intersection(mask_1, mask_2, priority=1):
    """
    This function removes the overlap between two masks on one of those two masks depending on the priority parameter.
    :param mask_1: First mask, with an intersecting section with the sectond one
    :param mask_2: Second mask, with an intersecting section with the first one
    :param priority: Tells which of the two arrays should be kept. Can only be 1 or 2.
    :type priority: int
    :return split_mask_1: First mask, minus the intersection if it doesn't have priority
    :return split_mask_2: Sectond mask, minus the intersection if it doesn't have priority
    """

    if priority not in [1, 2]:
        raise Exception("Parameter priority can only be 1 or 2")

    array_1 = mask_1.astype(np.bool)
    array_2 = mask_2.astype(np.bool)
    intersection = (array_1 & array_2).astype(np.uint8)

    if priority is 1:
        mask_2 = mask_2 - intersection
    if priority is 2:
        mask_1 = mask_1 - intersection

    return mask_1, mask_2
