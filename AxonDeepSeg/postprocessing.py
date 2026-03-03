""""
Tools for the FSLeyes plugin and other functions for manipulating masks.
"""
from skimage import measure, morphology, segmentation
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
import pandas as pd
from matplotlib import font_manager
from loguru import logger

from AxonDeepSeg import params
from AxonDeepSeg.ads_utils import convert_path, imwrite
import AxonDeepSeg.morphometrics.compute_morphometrics as compute_morphs

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

def generate_rotated_ellipse_points(center_x, center_y, semi_major, semi_minor, orientation, num_points=64):
    """
    Generate points along a rotated ellipse perimeter.
    
    :param center_x: X coordinate of ellipse center
    :param center_y: Y coordinate of ellipse center
    :param semi_major: Semi-major axis length
    :param semi_minor: Semi-minor axis length
    :param orientation: Rotation angle in radians (0th axis to major axis)
    :param num_points: Number of points to sample along the ellipse
    :return: List of (x, y) tuples representing the ellipse perimeter
    """
    points = []
    orientation = -orientation  # Negate to match image coordinate system
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        # Unrotated ellipse points
        x_ellipse = semi_minor * np.cos(t)
        y_ellipse = semi_major * np.sin(t)
        
        # Rotate by orientation angle
        cos_angle = np.cos(orientation)
        sin_angle = np.sin(orientation)
        x_rotated = x_ellipse * cos_angle - y_ellipse * sin_angle
        y_rotated = x_ellipse * sin_angle + y_ellipse * cos_angle
        
        # Translate to center
        x_final = center_x + x_rotated
        y_final = center_y + y_rotated
        points.append((x_final, y_final))
    
    return points

def generate_diameter_overlay(stats_dataframe, image_shape, pixel_size, line_width=2, axon_shape='circle'):
    """
    Generate an overlay image with concentric circles or ellipses for each axon.
    For circles: each axon has two circles: one with axon_diameter and one with axon_diameter + 2*myelin_thickness.
    For ellipses: each axon has two ellipses with correct major/minor axes and orientation based on eccentricity and orientation columns.
    
    :param stats_dataframe: DataFrame containing axon morphometrics with columns: x0, y0, axon_diam, myelin_thickness, eccentricity, orientation, fiber_eccentricity, fiber_orientation
    :param image_shape: Tuple (height, width) of the image
    :param pixel_size: Pixel size in micrometers (used to convert diameters back to pixels)
    :param line_width: Width of the outlines in pixels (default: 2)
    :param axon_shape: Shape of the axons ('circle' or 'ellipse').
    :return: numpy array with white circle/ellipse outlines on black background
    """    
    
    overlay = Image.new('L', (image_shape[1], image_shape[0]), color=0)
    draw = ImageDraw.Draw(overlay)
    
    compute_bbox_circle = lambda center_x, center_y, radius: (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius
    )
    
    # Iterate through each axon in the stats_dataframe
    for idx, row in stats_dataframe.iterrows():
        x0 = row['x0']
        y0 = row['y0']
        
        if pd.isna(x0) or pd.isna(y0) or pd.isna(row['axon_diam']):
            logger.debug(f"Skipping axon {idx}: missing centroid or axon_diam")
            continue
        
        # Convert from micrometers to pixels
        axon_diam_px = row['axon_diam'] / pixel_size
        axon_radius_px = axon_diam_px / 2
        
        if axon_shape == 'ellipse':
            # For ellipse mode, axon_diam is the minor axis
            # Calculate semi-major axis from eccentricity using: a = b / sqrt(1 - e^2)
            if pd.isna(row['eccentricity']):
                logger.debug(f"Skipping axon {idx}: missing eccentricity for ellipse mode")
                continue
            
            e = row['eccentricity']
            # Avoid division by zero and invalid eccentricity values
            if e >= 1.0 or e < 0.0:
                logger.debug(f"Skipping axon {idx}: invalid eccentricity value {e}")
                continue
            
            # Get orientation if available, default to 0
            orientation = row['orientation'] if not pd.isna(row['orientation']) else 0.0
            
            semi_minor_axis_px = axon_radius_px  # This is b
            # Calculate semi-major axis: a = b / sqrt(1 - e**2)
            semi_major_axis_px = semi_minor_axis_px / np.sqrt(1 - e**2)
            
            # Generate and draw inner ellipse (axon)
            inner_points = generate_rotated_ellipse_points(x0, y0, semi_major_axis_px, semi_minor_axis_px, orientation)
            draw.polygon(inner_points, outline=255, width=line_width)
            
            # Draw outer ellipse (axon + myelin) if myelin_thickness is available
            if not pd.isna(row['myelin_thickness']):
                myelin_thickness_px = row['myelin_thickness'] / pixel_size
                outer_semi_minor_px = semi_minor_axis_px + myelin_thickness_px
                
                # Use fiber_eccentricity and fiber_orientation if available
                if not pd.isna(row['fiber_eccentricity']) and not pd.isna(row['fiber_orientation']):
                    fiber_e = row['fiber_eccentricity']
                    # Avoid division by zero and invalid eccentricity values
                    if 0.0 <= fiber_e < 1.0:
                        # Calculate semi-major axis from fiber eccentricity and outer minor axis
                        outer_semi_major_px = outer_semi_minor_px / np.sqrt(1 - fiber_e**2)
                        fiber_orientation = row['fiber_orientation']
                    else:
                        logger.debug(f"Skipping outer ellipse for axon {idx}: invalid fiber_eccentricity value {fiber_e}")
                        continue
                else:
                    logger.debug(f"Skipping outer ellipse for axon {idx}: missing fiber_eccentricity or fiber_orientation")
                    continue
                
                outer_points = generate_rotated_ellipse_points(x0, y0, outer_semi_major_px, outer_semi_minor_px, fiber_orientation)
                draw.polygon(outer_points, outline=255, width=line_width)
        elif axon_shape == 'circle':
            # Draw inner circle (axon diameter)
            x_min, y_min, x_max, y_max = compute_bbox_circle(x0, y0, axon_radius_px)
            
            draw.ellipse(
                [(x_min, y_min), (x_max, y_max)],
                outline=255,
                width=line_width
            )
            
            # Draw outer circle (axon + myelin) if myelin_thickness is available
            if not pd.isna(row['myelin_thickness']):
                myelin_thickness_px = row['myelin_thickness'] / pixel_size
                outer_radius_px = axon_radius_px + myelin_thickness_px
                
                x_min_outer, y_min_outer, x_max_outer, y_max_outer = compute_bbox_circle(x0, y0, outer_radius_px)
                
                draw.ellipse(
                    [(x_min_outer, y_min_outer), (x_max_outer, y_max_outer)],
                    outline=255,
                    width=line_width
                )
    
    # Convert PIL image to numpy array
    overlay_array = np.array(overlay, dtype=np.uint8)
    
    return overlay_array

def save_diameter_overlay(overlay_array, output_path):
    """
    Save the diameter overlay array as an image file.
    
    :param overlay_array: numpy array with the diameter overlay
    :param output_path: Path where the overlay image will be saved
    :return: None
    """    
    if overlay_array is None:
        logger.debug("Diameter overlay is None, skipping save.")
        return
    
    output_path = convert_path(output_path)
    
    try:
        imwrite(str(output_path), overlay_array)
        logger.info(f"Diameter overlay saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save diameter overlay to {output_path}: {e}")
