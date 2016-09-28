# This explains how to train, visualize and segment on a new image

# download + unzip example data
# https://www.dropbox.com/s/juybkrzzafgxkuu/victor.zip?dl=1

# input data to build the training set
path_data = './victor/data'

# output path of training data path
path_training = './trainingset'

# output path for trained U-Net
path_model = './models/Unet_new'
# optional input path of a model to initialize the training
path_model_init = './models/Unet_init'

# input to train the mrf. Here we don't use all data for the MRF for faster results and redundancy.
path_mrf_training = ['./victor/data/data2', './victor/data/data4']
path_mrf = './models/mrf'

# input to segment an image
path_my_data = './victor/my_data/data4'

# Generate training path
from AxonDeepSeg.learning.data_construction import build_data
build_data(path_data, path_training, trainRatio=0.80)

# OPTION 1: Local Train
# Training the U-Net from a path_training
from AxonDeepSeg.learn_model import learn_model
learn_model(path_training, path_model, learning_rate=0.005)
#Initialize the training
learn_model(path_training, path_model,path_model_init, learning_rate=0.002)

# OPTION 2: Train on GPU
# copy training data + model (in case you start from an existing model) onto neuropoly@ssh
scp -r AxonDeepSeg neuropoly@bireli.neuro.polymtl.ca:
scp -r path_training neuropoly@bireli.neuro.polymtl.ca:my_project #path on bireli : path_bireli_training
scp -r path_model_init neuropoly@bireli.neuro.polymtl.ca:my_project/models # path on bireli : path_bireli_model_init
# Connect to bireli using ssh neuropoly@ssh
git clone https://github.com/neuropoly/axondeepseg.git
cd axondeepseg/AxonDeepSeg
# sub-option1: if you don't have an initial model:
python learn_model.py -p path_bireli_training -m path_bireli_model -lr 0.0005 # result : path_bireli_model
# sub-option2: if you do have an initial model:
python learn_model.py -p path_bireli_training -m path_bireli_model -m_init path_bireli_model_init  -lr 0.0005
# In a local Terminal window, visualize to visualize the training performances:
scp -r neuropoly@bireli.neuro.polymtl.ca:my_project/models/model path_model

#Visualization of the training
from AxonDeepSeg.evaluation.visualization import visualize_learning
# OPTION 1 : if you do not have an initial model
visualize_learning(path_model)
# OPTION 2 if you do not have an initial model
visualize_learning(path_model, path_model_init)

# Training the MRF from the paths_training
from AxonDeepSeg.mrf import learn_mrf
learn_mrf(path_mrf_training, path_model, path_mrf)

#----------------------Axon segmentation with a trained model and trained mrf---------------------#
from AxonDeepSeg.apply_model import axon_segmentation
# Ground Truth need to be in path_my_data
axon_segmentation(path_my_data, path_model, path_mrf)
# Outputs :
# Axon segmentation mask AxonSeg.jpeg
# Segmentation array after and before mrf in results.pkl

#----------------------Myelin segmentation from axon segmentation--------------------#
from AxonDeepSeg.apply_model import myelin
# Axon_segmentation() has to be already runned
myelin(path_my_data)
# Outputs in (path_my_data) :
# Myelin segmentation mask, MyelinSeg.jpg
# Metrics evalution of the segmentation, axonlist.mat

#----------------------Axon and Myelin segmentation--------------------#
from AxonDeepSeg.apply_model import pipeline
pipeline(path_my_data, path_model, path_mrf)
# Outputs :
# Outputs from myelin() and axon_segmentation()

#----------------------Visualization of the results--------------------#
from AxonDeepSeg.evaluation.visualization import visualize_results
visualize_results(path_my_data)
# Outputs :
# Plots of the axon segmentation and the myelin segmentation
# if my_data does have a groundtruth : report_results.txt in path_my_data

