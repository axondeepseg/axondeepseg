""""
Tools for the FSLeyes plugin and other functions for manipulating masks.
"""
from skimage import measure, morphology, segmentation
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
from AxonDeepSeg import params
import AxonDeepSeg.morphometrics.compute_morphometrics as compute_morphs
from matplotlib import font_manager

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

def fill_myelin_holes(myelin_array, max_area_fraction=0.1):
    """
    This function uses the fill_small_holes function from scikit-image to fill closed myelin objects with the axon mask.
    :param myelin_array: the binary array corresponding to the myelin mask
    :param max_area_factor (optional): fraction of the image size which will determine the maximum area that a hole will
    be considered an axon. The value must be between 0 and 1. Default: 0.1 
    :return: the binary axon array corresponding to the axon mask after the floodfill
    """
    # Get the dimensions of the image
    image_dims = myelin_array.shape

    # Determine the maximum axon area in pixels
    maximum_axon_area = max_area_fraction * image_dims[0] * image_dims[1]

    #Fill the myelin array
    filled_array = morphology.remove_small_holes(myelin_array.astype(bool), area_threshold=maximum_axon_area)
    filled_array = filled_array.astype(np.uint8)

    #Extract the axon array
    axon_extracted_array = filled_array-myelin_array
    return axon_extracted_array

def remove_intersection(mask_1, mask_2, priority=1, return_overlap=False):
    """
    This function removes the overlap between two masks on one of those two masks depending on the priority parameter.
    :param mask_1: First mask, with an intersecting section with the sectond one. The mask must contain values of 1 or 0
    only.
    :param mask_2: Second mask, with an intersecting section with the first one. The mask must contain values of 1 or 0
    only.
    :param priority (optional): Tells which of the two arrays should be kept. Can only be 1 or 2.
    :type priority: int
    :param return_overlap (optional): If set to True, the overlap mask will be returned.
    :type return_overlap: bool
    :return split_mask_1: First mask, minus the intersection if it doesn't have priority
    :return split_mask_2: Sectond mask, minus the intersection if it doesn't have priority
    :return return_overlap: A mask corresponding to the intersection of the two masks
    """

    if priority not in [1, 2]:
        raise Exception("Parameter priority can only be 1 or 2")

    array_1 = mask_1.astype(bool)
    array_2 = mask_2.astype(bool)
    intersection = (array_1 & array_2).astype(np.uint8)

    if priority == 1:
        mask_2 = mask_2 - intersection
    if priority == 2:
        mask_1 = mask_1 - intersection

    if return_overlap is True:
        return mask_1, mask_2, intersection
    else:
        return mask_1, mask_2

def generate_axon_numbers_image(centroid_index, x0_array, y0_array, image_size, mean_axon_diameter_in_pixels=None):
    """
    This function generates an image where the numbers in centroid_index are at their corresponding location specified
    by their coordinates (x0, y0)
    :param centroid_index: The array containing the axon numbers.
    :type centroid_index: numpy array
    :param x0_array: X coordinate of the centroids
    :type x0_array: numpy array
    :param y0_array: Y coordinate of the centroids
    :type y0_array: numpy array
    :param image_size: the size of the image
    :type image_size: tuple
    :param mean_axon_diameter_in_pixels (optional): the mean axon diameter of the axon mask. If this parameter
    is passed, the font size will be determined based on it. Otherwise, the image size will be used to determine
    the font size.
    :return: the binary image with the numbers at their corresponding coordinate.
    """

    # Create an empty image which will contain the binary image
    number_image = Image.new(mode='L', size=image_size, color=0)
    draw = ImageDraw.Draw(number_image)

    # Use a font from the matplotlib package
    # The size of the font depends on the dimensions of the image, should be at least 10
    # If the the mean_axon_diameter is specified, use it to determine the font_size
    font_path = font_manager.findfont("DejaVu Sans")
    if mean_axon_diameter_in_pixels is None:
        font_size = max(int(sum(image_size) * 0.5 * 0.01), 10)
    else:
        font_size = max(int(mean_axon_diameter_in_pixels / 3), 10)
    font = ImageFont.truetype(font=font_path, size=font_size)

    # Fill the image with the numbers at their corresponding coordinates
    for i in range(centroid_index.size):
        draw.text(xy=(x0_array[i] - font_size/2, y0_array[i] - font_size/2),
                  text=str(centroid_index[i]), font=font, fill=params.intensity['binary'])

    # Transform the image into a numpy array
    image_array = np.asarray(number_image)

    return image_array.astype(np.uint8)

def generate_and_save_colored_image_with_index_numbers(filename, axonmyelin_image_path, index_image_array):
    """
    This function generates an RGB image with the axons in blue and myelin in red with the axon indexes overlayed on
    top of the image.
    :param filename: The name of the colored image that will be saved
    :type filename: String or Path
    :param axonmyelin_image_path: The path a grayscale axonmyelin mask
    :type axonmyelin_image_path: String or Path
    :param index_image_array: The array containing the index image
    :type index_image_array: 2D Numpy array
    """
    seg = Image.open(axonmyelin_image_path)
    index_image = Image.fromarray(index_image_array)
    colored_image = ImageOps.colorize(seg, black="black", white="blue", mid="red",
                                      blackpoint=params.intensity["background"],
                                      whitepoint=params.intensity["axon"],
                                      midpoint=params.intensity["myelin"])
    colored_image.paste(index_image, mask=index_image)
    colored_image.save(filename)

def remove_axons_at_coordinates(im_axon, im_myelin, x0s, y0s):
    """
    Removes axonmyelin objects at the (x, y) coordinates passed as parameters
    :param im_axon: Array: axon binary mask
    :param im_myelin: Array: myelin binary mask
    :param x0s: list of ints/floats: X coordinates of the axonmyelin object to be removed
    :param y0s: list of ints/floats: Y coordinates of the axonmyelin object to be removed
    :return: axon and myelin array with the axonmyelin objects removed
    """
    im_axon, im_myelin = remove_intersection(im_axon, im_myelin)
    watershed_seg = compute_morphs.get_watershed_segmentation(im_axon, im_myelin)

    #perform a floodfill at the coordinates passed in parameters
    for i in range(len(x0s)):
        value_to_remove = np.iinfo(np.int16).max
        watershed_seg = segmentation.flood_fill(watershed_seg, (int(y0s[i]), int(x0s[i])),  value_to_remove)
        # The axons to remove now have a value of 65535 (value_to_remove) on the watershed_seg

    axons_to_remove = (watershed_seg == value_to_remove).astype(np.uint8) # Binairy mask of the location of
                                                                          # the axons to remove
    original_axonmyelin_array = im_axon + im_myelin
    new_axonmyelin_array, _ = remove_intersection(original_axonmyelin_array, axons_to_remove, priority=2)

    axon_array = (im_axon & new_axonmyelin_array).astype(np.uint8)
    myelin_array = (im_myelin & new_axonmyelin_array).astype(np.uint8)
    return axon_array, myelin_array
