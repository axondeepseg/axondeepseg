# Tools to colorize instance maps with arbitrary color palettes

import itertools
import random
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from loguru import logger


class color_generator(object):
    '''Generator that yields the next color given a color palette.'''

    def __init__(self, colors, mem_length=30, tolerance=30, seed=42):
        '''
        Initialize the object given a color palette.
        :param colors:      List of colors in RGB format. e.g. [R, G, B]
        :param mem_length   Size of the buffer of generated colors
        :param tolerance    Max difference between colors in the buffer 
                            (computed by summing R/G/B channel differences)
        :param seed         Controls the pseudo-random number generator seed
        '''
        if len(colors) < 4:
            raise ValueError("Please provide 4 colors or more.")
        self.first_color = colors[0]

        random.seed(seed)
        
        # cast the color list into a cyclic iterator
        self.colors = itertools.cycle(colors)
        self.current_color = next(self.colors)

        self.tolerance = tolerance
        
        # df to store the colors already generated
        self.generated = deque([], maxlen=mem_length)


    def __iter__(self):
        return self

    def __next__(self):
        '''Returns a value between 2 colors from the input palette.'''
        while True:
            c1 = self.current_color
            c2 = next(self.colors)
            # check for end of a cycle
            if c2 is self.first_color:
                c1 = c2
                c2 = next(self.colors)

            #generate a color inbetween c1 and c2
            color = (
                random.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
                random.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
                random.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
            )
            self.current_color = c2

            # first iteration: directly return the color
            if not self.generated:
                self.generated.append(color)
                return color

            # check if color is too close to the ones inside the memory
            memory = np.array(self.generated)
            diff = abs(memory - list(color))
            cumulative_diff = diff.sum(axis=1)
            if np.all(cumulative_diff > self.tolerance):
                self.generated.append(color)
                return color


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
            (217, 237, 146),
            (118, 200, 147),
            (22, 138, 173),
            (24, 78, 119),
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
        pt_list = zip(instance[1], instance[0])
        for pt in pt_list:
            draw.point(pt, color)
    
    return np.asarray(colorized)