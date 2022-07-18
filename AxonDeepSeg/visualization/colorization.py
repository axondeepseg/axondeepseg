# Tools to colorize instance maps with arbitrary color palettes

import itertools
import numpy as np
import pandas as pd
from skimage.future import graph
from random import randint

from loguru import logger


class color_generator(object):
    '''Generator that yields the next color given a color palette.'''

    def __init__(self, colors):
        '''
        Initialize the object given a color palette.
        :param colors:  List of colors in RGB format. e.g. [r, g, b]
        '''
        if len(colors) < 4:
            raise ValueError("Please provide 4 colors or more.")
        if [0, 0, 0] in colors:
            raise ValueError("Please avoid using black (background color).")
        
        # cast the color list into a cyclic iterator
        self.colors = itertools.cycle(colors)
        self.current_color = next(self.colors)
        
        # dataframe to store the colors already generated
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

            # check if already generated; if not, return color
            color_to_check = np.array(color)
            if not (self.generated_df == color_to_check).all(1).any():
                # flag color as already generated
                self.generated_df.loc[len(self.generated_df)] = color
                return list(color.values())
            
            self.current_color = c2



def colorize_instance_segmentation(instance_seg, image, colors=None):
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

        ]
    color_gen = color_generator(colors)

