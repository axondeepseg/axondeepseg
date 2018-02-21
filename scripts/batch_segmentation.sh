#!/bin/bash
#
# Batch that segments samples with AxonDeepSeg
# - 1. Pre-processing of samples if necessary
# - 2. Segmentation with AxonDeepSeg
# - 3. Cleaning of outputs if necessary
# - 4. Computation of morphometrics for each sample
# 
# The expected structure is a main folder (PATH_DATA) that contains different subfolders for each sample.
# Each subfolder should contain: 1) The image sample; 2) A txt file 'pixel_size_in_micrometer.txt' of the pixel size
# 
# Note: AxonDeepSeg needs to be installed first. For installation, please to to:
# https://neuropoly.github.io/axondeepseg/documentation.html#getting-started
#
# Aldo Zaimi <aldo.zaimi@polymtl.ca>
# 2018-02-20
#-------------------------------------------------------------------------------------------------------


# Parameters--------------------------------------------------------------------------------------------

PATH_ADS=" "  # path to AxonDeepSeg virtual environment
export PATH_DATA=" "  # path that contains the histology data

export MODALITY="TEM" # modality type of samples to segment (SEM or TEM)

# ------------------------------------------------------------------------------------------------------

# Activate virtual environment
source ${PATH_ADS}/bin/activate

# Go to data folder
cd ${PATH_DATA}

# Pre-process the input images to make them compatible with AxonDeepSeg -------------------------------
# Note: pre-processing includes: 
# - conversion to png if necessary
# - image manipulation if TEM model: use negative image (255-img) for current TEM model trained on the negative contrast
# 										 if SEM model: no manipulation for current SEM model

python << END

import imageio # JCA: add these as dependencies
import os

MODALITY = os.environ.get('MODALITY')
PATH_DATA = os.environ.get('PATH_DATA')

list_folders = [x[0] for x in os.walk(PATH_DATA)]

for x in list_folders[1:]:
	os.chdir(x)
	for root, dirs, files in os.walk("."):  
		for filename in files:
			if (filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".tif")):
				img = imageio.imread(filename)
				os.remove(filename)
				img_name, file_extension = os.path.splitext(filename)
				img_name = img_name + '.png'
				if MODALITY == "SEM":
					imageio.imwrite(img_name,img)
				if MODALITY == "TEM":
					imageio.imwrite(img_name,255 - img)

			if filename.endswith(".png"):
				img = imageio.imread(filename)
				if MODALITY == "TEM":
					imageio.imwrite(filename,255 - img)
	os.chdir("../")
END

# Launch segmentation for each subfolder ---------------------------------------------------------------
echo "* * * Launching segmentation of samples in folders. * * *"

dirs=(${PATH_DATA}/*/)
for dir in "${dirs[@]}"
do
		echo "* * * Processing subfolder $dir * * *"
		axondeepseg -t ${MODALITY} -i $dir

		echo "* * * Cleaning unecessary outputs. * * *"
		cd $dir
		rm "axon_mask.png"
		rm "myelin_mask.png"
		cd ..
done

# More cleaning if necessary (if sample images were changed for AxonDeepSeg, revert back to original) --

python << END

import imageio
import os

MODALITY = os.environ.get('MODALITY')
PATH_DATA = os.environ.get('PATH_DATA')

list_folders = [x[0] for x in os.walk(PATH_DATA)]

for x in list_folders[1:]:
	os.chdir(x)
	for root, dirs, files in os.walk("."):  
		for filename in files:
			if (filename.endswith(".png") and not(filename.endswith("segmented.png"))):
					img = imageio.imread(filename)
					if MODALITY == "TEM":
						imageio.imwrite(filename,255 - img)
	os.chdir("../")

END

# Automatically compute the morphometrics of segmented sample -----------------------------------------
echo "* * * Computing morphometrics of segmented samples. * * *"

python << END

import imageio
import os
import numpy as np

from AxonDeepSeg.morphometrics.compute_morphometrics import *

MODALITY = os.environ.get('MODALITY')
PATH_DATA = os.environ.get('PATH_DATA')

list_folders = [x[0] for x in os.walk(PATH_DATA)]

for x in list_folders[1:]:
	os.chdir(x)
	for root, dirs, files in os.walk("."):  
		for filename in files:    	
			if filename.endswith(".png"):  
				img = imageio.imread(filename)
			if filename.endswith("segmented.png"):
					pred = imageio.imread(filename)
					pred_axon = pred > 200
					pred_myelin = np.logical_and(pred >= 50, pred <= 200)
					path_folder = os.getcwd()
					stats_array = get_axon_morphometrics(pred_axon,path_folder)
					save_axon_morphometrics(path_folder,stats_array)
					display_axon_diameter(img,path_folder,pred_axon,pred_myelin)
					aggregate_metrics = get_aggregate_morphometrics(pred_axon,pred_myelin,path_folder)
					write_aggregate_morphometrics(path_folder,aggregate_metrics)
	os.chdir("../")

END

# Deactivate virtual environment -----------------------------------------------------------------------
echo "* * * Processing finished. Deactivating virtual environment now. * * *"
deactivate ${PATH_ADS}

