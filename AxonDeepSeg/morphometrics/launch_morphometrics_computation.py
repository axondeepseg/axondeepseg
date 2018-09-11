# coding: utf-8

import numpy as np
import math
import os
import imageio
from sys import platform as _platform
if _platform == "darwin": # Mac OSX
    import matplotlib as mpl
    mpl.use('TkAgg')
import matplotlib.pyplot as plt

from AxonDeepSeg.morphometrics.compute_morphometrics import *
import AxonDeepSeg.ads_utils


def launch_morphometrics_computation(path_img,path_prediction):
	'''
	This function is equivalent to the morphometrics_extraction notebook of AxonDeepSeg.
	It automatically performs all steps (computations, savings, displays,...) of the 
	morphometrics extraction of a given sample.
	:param path_img: path of the input image (microscopy sample)
	:param path_prediction: path of the segmented image (output of AxonDeepSeg)
	:return: none.
	'''
	try:
		# Read image
		img = imageio.imread(path_img)
	
		# Read prediction
		pred = imageio.imread(path_prediction)
	except (IOError, OSError) as e:
		print(("launch_morphometrics_computation: " + str(e)))
		raise
	else:

		# Get axon and myelin masks
		pred_axon = pred > 200
		pred_myelin = np.logical_and(pred >= 50, pred <= 200)

		# Get folder path
		path_folder, file_name = os.path.split(path_img)

		# Compute and save axon morphometrics
		stats_array = get_axon_morphometrics(pred_axon,path_folder)
		save_axon_morphometrics(path_folder,stats_array)

		# Generate and save displays of axon morphometrics
		display_axon_diameter(img,path_prediction,pred_axon,pred_myelin)

		# Compute and save aggregate morphometrics
		aggregate_metrics = get_aggregate_morphometrics(pred_axon,pred_myelin,path_folder)
		write_aggregate_morphometrics(path_folder,aggregate_metrics)

