# This explains how to train, visualize and segment on a new image

#-------------------------------- Variables ---------------------------#

# input to build the training set
path_data = '/Users/viherm/Desktop/CARS'
path_training = '/Users/viherm/Desktop/trainingset'

# input to train the U-Net
path_model = '/Users/viherm/Desktop/data/models/model_init'
path_model_new = '/Users/viherm/Desktop/data/models/model_new'

# input to train the mrf
paths_training = ['/Users/viherm/Desktop/CARS/data1','/Users/viherm/Desktop/CARS/data2', '/Users/viherm/Desktop/CARS/data3']
path_mrf = '/Users/viherm/Desktop/data/models/mrf'

# input to segment an image
path_my_data = '/Users/viherm/Desktop/data2segment/mydata'

# Generate training path
from AxonDeepSeg.learning.data_construction import build_data
build_data(path_data, path_training, trainRatio=0.80)

# OPTION 1: Train on CPU
# Training the U-Net from a path_training
from AxonDeepSeg.learn_model import learn_model
learn_model(path_training, path_model, learning_rate=0.005)
#Initialize the training
learn_model(path_training, path_model_new, path_model, learning_rate=0.002)
#Visualization of the training
from AxonDeepSeg.evaluation.visualization import visualize_learning
visualize_learning(path_model)

# OPTION 2: Train on GPU
# copy training data + model (in case you start from an existing model) onto neuropoly@ssh 
scp path_training neuropoly@bireli.neuro.polymtl.ca:my_project
scp path_model neuropoly@bireli.neuro.polymtl.ca:my_project
# Connect to bireli using ssh neuropoly@ssh
cd AxonSegmentation/AxonDeepSeg
# sub-option1: if you don't have an initial model:
python learn_model.py -p path_bireli_training -m path_bireli_model_new -lr 0.0005
# sub-option2: if you do have an initial model:
python learn_model.py -p path_bireli_training -m path_bireli_model_new -m_init path_bireli_model_init  -lr 0.0005
# In a local Terminal window, visualize to visualize the training performances:
scp -r path_bireli_model_new path_model_new

# Training the MRF from the paths_training
from AxonDeepSeg.mrf import learn_mrf
learn_mrf(paths_training, path_mrf)

#----------------------Axon segmentation with a trained model and trained mrf---------------------#
from AxonDeepSeg.apply_model import axon_segmentation
axon_segmentation(path_my_data, path_model, path_mrf)

#----------------------Myelin segmentation from Axon segmentation--------------------#
from AxonDeepSeg.apply_model import myelin
myelin(path_my_data)

#----------------------Axon and Myelin segmentation--------------------#
from AxonDeepSeg.apply_model import pipeline
pipeline(path_my_data,path_model,path_mrf)

#----------------------Visualization of the results--------------------#
from AxonDeepSeg.evaluation.visualization import visualize_results
visualize_results(path_my_data)





