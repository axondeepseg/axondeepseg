# Tools to colorize instance maps with arbitrary color palettes

import itertools
import numpy as np
import pandas as pd
from skimage.future import graph

from loguru import logger


class color_generator(object):
    '''Generator that yields the next color given a color palette'''

    def __init__(self, colors):
        '''
        Initialize the object given a color palette
        :param colors:  List of colors in RGB format. e.g. [r, g, b]
        '''
        if len(colors) < 4:
            raise ValueError("Please provide 4 colors or more.")
        if len(colors) != len(np.unique(colors)):
            raise ValueError("Please provide only unique colors.")
        
        # cast the color list into a cyclic iterator
        self.colors = itertools.cycle(colors)
        
        # dataframe to store the colors already generated
        generated_df = pd.DataFrame({"R": [], "G": [], "B": []})

    def __iter__(self):
        return self

    def __next__(self):
        #TODO: take 2 next colors
        #TODO: generate a color inbetween these 2
        #TODO: check if already generated; if not, return color



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

    logger.info("Computing Region Adjacency Graph (RAG)")
    rag = graph.RAG(instance_seg)
    rag.remove_node(0)
    for region in rag:
        print(f"Neighbors found: {list(rag.neighbors(region))}")