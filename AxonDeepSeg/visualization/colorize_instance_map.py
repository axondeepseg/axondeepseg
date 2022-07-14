# Tools to colorize instance maps with arbitrary color palettes

import itertools
import numpy as np

def color_generator(colors):
    '''a generator that yields the next color given a color palette'''
    if len(colors) < 4:
        raise ValueError("Please provide 4 colors or more.")
    if len(colors) != len(np.unique(colors)):
        raise ValueError("Please provide only unique colors.")
    
    # cast the color list into a cyclic iterator
    colors = itertools.cycle(colors)
    while True:
        yield next(colors)

def colorize_instance_segmentation(instance_seg, colors=None):
    '''
    Colorizes an instance segmentation such that adjacent objects are 
    never of the same color (4 colors or more required)
    :param instance_seg:    instance segmentation to colorize
    :param colors:          color palette to use for the colorization
    :return:                colorized instance segmentation
    '''

    if colors is None:
        colors = [

        ]
    colormap = color_generator(colors)