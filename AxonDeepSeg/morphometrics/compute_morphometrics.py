# coding: utf-8

import numpy as np
from skimage import io
from scipy.misc import imread, imsave
import os
import imageio
import json
from skimage import transform
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from shutil import copy
import math
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.testing.segmentation_scoring import *
from skimage import measure
from skimage.measure import regionprops


def get_pixelsize(path_pixelsize_file):
	'''
	:param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
	:return: the pixel size value.
	'''

	text_file = open(path_pixelsize_file, "r")
	pixelsize = float(text_file.read())
	text_file.close()
	return pixelsize



def get_axon_morphometrics(pred_axon,path_folder):
	'''
	:param pred_axon: axon binary mask, output of axondeepseg
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
	np.save(os.path.join(path_folder,'axonlist.npy'),stats_array)


def load_axon_morphometrics(path_folder):
	'''
	:param path_folder: absolute path of folder containing the sample + the segmentation output
	:return: stats_array: list of dictionaries containing axon morphometrics
	'''
	stats_array = np.load(os.path.join(path_folder,'axonlist.npy'))
	return stats_array


def display_axon_diameter(img,path_prediction,pred_axon,pred_myelin):
	'''
	:param img: sample grayscale image (png)
	:param path_folder: absolute path of folder containing sample + segmentation
	:param aggregate_metrics: dictionary containing values of aggregate metrics
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
		
	# # Axon display
	# plt.figure(figsize=(12,9))
	# im = plt.imshow(axon_diam_display,cmap='hot')
	# plt.colorbar(im, fraction=0.03, pad=0.02)
	# plt.title('Axon display colorcoded with axon diameter in um',fontsize=12)           
	# plt.savefig(os.path.join(path_folder,'display_axon_diameter.png'))   
	
	# # Axon overlay on original image
	# plt.figure(figsize=(12,9))
	# plt.imshow(img, cmap='gray', alpha=0.8)
	# im = plt.imshow(axon_diam_display, cmap='hot', alpha=0.5)
	# plt.colorbar(im, fraction=0.03, pad=0.02)
	# plt.title('Axon overlay colorcoded with axon diameter in um',fontsize=12)
	# plt.savefig(os.path.join(path_folder,'overlay_axon_diameter.png')) 
	
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
	f = open(os.path.join(path_folder,'aggregate_morphometrics.txt'), 'w')
	f.write('aggregate_metrics: ' + repr(aggregate_metrics) + '\n')
	f.close()



