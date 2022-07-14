# Tools to colorize instance maps with arbitrary color palettes

import itertools
import numpy as np
from skimage.future import graph

from loguru import logger

def color_generator(colors):
    '''a generator that yields the next color given a color palette'''
    if len(colors) < 4:
        raise ValueError("Please provide 4 colors or more.")
    if len(colors) != len(np.unique(colors)):
        raise ValueError("Please provide only unique colors.")
    #TODO: WARN IF "BLACK" WAS CHOSEN THAT BG IS BLACK
    
    # cast the color list into a cyclic iterator
    colors = itertools.cycle(colors)
    while True:
        yield next(colors)

def colorize_instance_segmentation(instance_seg, image, colors=None):
    '''
    Colorizes an instance segmentation such that adjacent objects are 
    never of the same color (4 colors or more required). Note that the 
    background color is always black so one should avoid using it as an 
    input value.
    :param instance_seg:    instance segmentation map to colorize
    :param image:           image to colorize
    :param colors:          color palette to use for the colorization
    :return:                colorized instance segmentation
    '''

    if colors is None:
        colors = [

        ]
    color_gen = color_generator(colors)

    logger.info("Computing Region Adjacency Graph (RAG)")
    rag = graph.RAG(instance_seg)
    rag.remove_node(0)
    for region in rag:
        print(f"Neighbors found: {list(rag.neighbors(region))}")