# coding: utf-8

import numpy as np
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio
import json
from skimage import transform
from skimage.filters import gaussian
from sys import platform as _platform
if _platform == "darwin": # Mac OSX
    import matplotlib as mpl
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from shutil import copy
import math
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.testing.segmentation_scoring import *
from skimage import measure
from skimage.measure import regionprops
import AxonDeepSeg.ads_utils


def get_pixelsize(path_pixelsize_file):
	'''
	:param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
	:return: the pixel size value.
	'''
	try:
		with open(path_pixelsize_file, "r") as text_file:
			pixelsize = float(text_file.read())
	except IOError as e:
		print(("\nError: Could not open file \"{0}\" from "
			  "directory \"{1}\".\n".format(path_pixelsize_file, os.getcwd())))
		raise
	except ValueError as e:
		print(("\nError: Pixel size data in file \"{0}\" is not valid – must "
			   "be a plain text file with a single a numerical value (float) "
			   " on the fist line.".format(path_pixelsize_file)))
		raise
	else:
		return pixelsize


def get_axon_morphometrics(pred_axon,path_folder):
	'''
	:param pred_axon: axon binary mask, output of axondeepseg
	:param path_folder: absolute path of folder containing pixel size file
	:return: list of dictionaries containing for each axon, various morphometrics
	'''

	# Array for keeping axon-wise metrics
	stats_array = np.empty(0)

	# Label each axon object
	labels = measure.label(pred_axon)
	axon_objects = regionprops(labels)

	# Get axon morphometrics of interest
	for props in axon_objects:
	
		# Centroid
		y0, x0 = props.centroid
		# Solidity
		solidity = props.solidity
		# Eccentricity
		eccentricity = props.eccentricity
		# Axon equivalent diameter in micrometers
		axon_diam = (props.equivalent_diameter)*get_pixelsize(os.path.join(path_folder,'pixel_size_in_micrometer.txt'))
		# Axon orientation angle
		orientation = props.orientation
	
		# Add metrics to list of dictionaries
		stats = {'y0': y0, 'x0': x0, 'axon_diam': axon_diam, 'solidity':solidity,'eccentricity': eccentricity,
				 'orientation':orientation}
		stats_array = np.append(stats_array, [stats], axis=0)
		
	return stats_array


def save_axon_morphometrics(path_folder,stats_array):
	'''
	:param path_folder: absolute path of folder containing the sample + the segmentation output
	:param stats_array: list of dictionaries containing axon morphometrics
	:return: nothing
	'''
	try:
		np.save(os.path.join(path_folder,'axonlist.npy'),stats_array)
	except IOError as e:
		print(("\nError: Could not save file \"{0}\" in "
			  "directory \"{1}\".\n".format('axonlist.npy', path_folder)))
		raise


def load_axon_morphometrics(path_folder):
	'''
	:param path_folder: absolute path of folder containing the sample + the segmentation output
	:return: stats_array: list of dictionaries containing axon morphometrics
	'''
	try:
		stats_array = np.load(os.path.join(path_folder,'axonlist.npy'))
	except IOError as e:
		print(("\nError: Could not load file \"{0}\" in "
			  "directory \"{1}\".\n".format('axonlist.npy', path_folder)))
		raise
	else:
		return stats_array


def display_axon_diameter(img,path_prediction,pred_axon,pred_myelin):
	'''
	:param img: sample grayscale image (png)
	:param path_prediction: full path to the segmented file (*_seg-axonmyelin.png) from axondeepseg segmentation output
	:param pred_axon: axon mask from axondeepseg segmentation output
	:param pred_myelin: myelin mask from axondeepseg segmentation output
	:return: nothing
	'''   
	path_folder, file_name = os.path.split(path_prediction)
	tmp_path = path_prediction.split('_seg-axonmyelin')

	stats_array = get_axon_morphometrics(pred_axon,path_folder)
	axon_diam_list = [d['axon_diam'] for d in stats_array]
	axon_diam_array = np.asarray(axon_diam_list)
	axon_iter = np.arange(np.size(axon_diam_array))

	labels = measure.label(pred_axon)
	axon_diam_display = a = np.zeros((np.shape(labels)[0], np.shape(labels)[1]))

	for pix_x in np.arange(np.shape(labels)[0]):
		for pix_y in np.arange(np.shape(labels)[1]):
			if labels[pix_x,pix_y] != 0:
				axon_diam_display[pix_x,pix_y] = axon_diam_array[labels[pix_x,pix_y]-1]
	
	# Axon overlay on original image + myelin display (same color for every myelin sheath)
	plt.figure(figsize=(12,9))
	plt.imshow(img, cmap='gray', alpha=0.8)
	plt.imshow(pred_myelin,cmap='gray', alpha=0.3)
	im = plt.imshow(axon_diam_display, cmap='hot', alpha=0.5)
	plt.colorbar(im, fraction=0.03, pad=0.02)
	plt.title('Axon overlay (colorcoded with axon diameter in um) and myelin display',fontsize=12)
	plt.savefig(tmp_path[0] + '_map-axondiameter.png')

def get_aggregate_morphometrics(pred_axon, pred_myelin, path_folder):
	'''
	:param pred_axon: axon mask from axondeepseg segmentation output
	:param pred_myelin: myelin mask from axondeepseg segmentation output
	:param path_folder: absolute path of folder containing pixel size file
	:return: aggregate_metrics: dictionary containing values of aggregate metrics
	'''
	
	# Compute AVF (axon volume fraction) = area occupied by axons in sample
	avf = np.count_nonzero(pred_axon)/float((pred_axon.size))
	# Compute MVF (myelin volume fraction) = area occupied by myelin sheaths in sample
	mvf = np.count_nonzero(pred_myelin)/float((pred_myelin.size))
	
	# Estimate aggregate g-ratio = sqrt(1/(1+MVF/AVF))
	gratio = math.sqrt(1/(1+(float(mvf)/float(avf))))
	
	# Get individual axons metrics and compute mean axon diameter
	stats_array = get_axon_morphometrics(pred_axon,path_folder)
	axon_diam_list = [d['axon_diam'] for d in stats_array]
	mean_axon_diam = np.mean(axon_diam_list)
	
	# Estimate mean myelin diameter (axon+myelin diameter) by using aggregate g-ratio = mean_axon_diam/mean_myelin_diam
	mean_myelin_diam = mean_axon_diam/gratio
	
	# Estimate mean myelin thickness = mean_myelin_radius - mean_axon_radius
	mean_myelin_thickness = (float(mean_myelin_diam)/2) - (float(mean_axon_diam)/2)
	
	# Compute axon density (number of axons per mm2)
	y, x = pred_axon.shape
	img_area_mm2 = float(pred_axon.size)*get_pixelsize(os.path.join(path_folder,'pixel_size_in_micrometer.txt'))*get_pixelsize(os.path.join(path_folder,'pixel_size_in_micrometer.txt'))/(float(1000000))
	axon_density_mm2 = float(len(axon_diam_list))/float(img_area_mm2)
	
	# Create disctionary to store aggregate metrics
	aggregate_metrics = {'avf': avf, 'mvf': mvf, 'gratio': gratio, 'mean_axon_diam': mean_axon_diam,
						 'mean_myelin_diam': mean_myelin_diam,'mean_myelin_thickness': mean_myelin_thickness,
						 'axon_density_mm2': axon_density_mm2}

	return aggregate_metrics


def write_aggregate_morphometrics(path_folder,aggregate_metrics):
	'''
	:param path_folder: absolute path of folder containing sample + segmentation
	:param aggregate_metrics: dictionary containing values of aggregate metrics
	:return: nothing
	'''   

	try:
		with open(os.path.join(path_folder,'aggregate_morphometrics.txt'), 'w') as text_file:
			text_file.write('aggregate_metrics: ' + repr(aggregate_metrics) + '\n')
	except IOError as e:
		print(("\nError: Could not save file \"{0}\" in "
			  "directory \"{1}\".\n".format('aggregate_morphometrics.txt', path_folder)))
		raise
