# Tools to colorize instance maps with arbitrary color palettes

import itertools

import numpy as np
import pandas as pd
from skimage.future import graph
from random import randint
from PIL import Image, ImageDraw

from loguru import logger


class color_generator(object):
    '''Generator that yields the next color given a color palette.'''

    def __init__(self, colors, mem_length=50, tolerance=30):
        '''
        Initialize the object given a color palette.
        :param colors:      List of colors in RGB format. e.g. [R, G, B]
        :param mem_length   Size of the buffer of generated colors
        :param tolerance    Max difference between colors in the buffer 
                            (computed by summing R/G/B channel differences)
        '''
        if len(colors) < 4:
            raise ValueError("Please provide 4 colors or more.")
        
        # cast the color list into a cyclic iterator
        self.colors = itertools.cycle(colors)
        self.current_color = next(self.colors)

        self.tailsize = mem_length
        self.tolerance = tolerance
        
        # df to store the colors already generated
        self.generated_df = pd.DataFrame({"R": [], "G": [], "B": []})

    def __iter__(self):
        return self

    def __next__(self):
        '''Returns a value between 2 colors from the input palette.'''
        while True:
            c1 = self.current_color
            c2 = next(self.colors)

            #generate a color inbetween c1 and c2
            color = {
                "R": randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
                "G": randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
                "B": randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
            }
            self.current_color = c2

            # check if color is too close to the ones inside the memory
            memory = self.generated_df.tail(self.tailsize)
            diff = abs(memory - color)
            cumulative_diff = diff["R"] + diff["G"] + diff["B"]
            if not (cumulative_diff < self.tolerance).any():
                self.generated_df.loc[len(self.generated_df)] = color
                return tuple(color.values())


def colorize_instance_segmentation(instance_seg, colors=None):
    '''
    Colorizes an instance segmentation such that adjacent objects are 
    never of the same color (4 colors or more required). Note that the 
    background color is always black so one should avoid using it as an 
    input value.
    :param instance_seg:    instance segmentation map to colorize
    :param image:           image to colorize
    :param colors:          color palette to use for the colorization in
                            RGB format like so: [R, G, B]
    :return:                colorized instance segmentation
    '''

    if colors is None:
        colors = [
            (0, 128, 128),
            (153, 50, 204),
            (0, 255, 127),
            (139, 0, 0),
        ]
    color_gen = color_generator(colors)

    # background color isn't an instance
    nb_unique_instances = len(np.unique(instance_seg)) - 1
    logger.info(f"Colorizing {nb_unique_instances} instances.")
    
    colorized = Image.fromarray(instance_seg)
    colorized = colorized.convert('RGB')
    draw = ImageDraw.Draw(colorized)

    for i in range(nb_unique_instances):
        instance_id = i + 1
        color = next(color_gen)
        instance = np.where(instance_seg == instance_id)
        px_list = zip(instance[1], instance[0])
        for pt in px_list:
            draw.point(pt, color)
    
    return np.asarray(colorized)