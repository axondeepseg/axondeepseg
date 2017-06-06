# AxonDeepSeg : Axon Deep Segmentation Toolbox
Segment axon and myelin from microscopy data. Written in Python.
Methods based on deep convolutional neural networks.
The U-Net from ([Ronneberger et al.](https://arxiv.org/abs/1505.04597)) is used as a pixel classifier model.
Then an MRF should be applied for the post-processing. # NOT WORKING FOR NOW :scripts in older branches.

The resulting axon segmentation mask feeds a myelin segmentation model ([Zaimi et al.](http://journal.frontiersin.org/article/10.3389/fninf.2016.00037/full))

<img src="https://github.com/neuropoly/axondeepseg/blob/master/doc/schema.jpg" width="600px" align="middle" />

## Installation
### Prerequisites : 
  * **Python 2.7** : https://www.python.org/download/releases/2.7/
  * **PIP** : https://pip.pypa.io/en/stable/installing/
  * **Tensorflow 1.1** (GPU enabled, Python 2.7) : https://www.tensorflow.org/install/install_linux#installing_with_virtualenv
  * **OpenCV** (Python 2.7) : http://opencv.org/
  
### Prerequisites only for myelin segmentation :
  * **Matlab** : https://www.mathworks.com/products/matlab/
  * **AxonSeg (Matlab)** : https://github.com/neuropoly/axonseg

### AxonDeepSeg :

  1. clone repository: ```git clone https://github.com/neuropoly/axondeepseg.git```
  2. cd into axondeepseg, then run: ```python set_config.py -p PATH_AXONSEG -m PATH_MATLAB``` (PATH_AXONSEG contains the AxonSeg Toolbox)
  3. run: ```python setup.py install```

Example dataset:
https://www.dropbox.com/s/juybkrzzafgxkuu/victor.zip?dl=1

##  File Structure

`
axondeepseg
-- AxonDeepSeg/
---- Files related to the algorithm
-- data/
---- dataset_name1/
------ training/
-------- Train/
---------- training images.png
-------- Validation/
---------- validation images.png
------ testing/
-------- testing image.png
-------- pixel_size_in_micrometer.txt
-------- ...
------ raw/
-------- raw images.png
---- dataset_name2/
---- ...
-- models/
---- model1/
------ train/
-------- summary for model1 for train set
------ validation/
-------- summary for model1 for validation set
------ report.txt
------ model.ckpt.meta
------ ...
---- model2/
---- ...
`


### N.B.
* the image needs to have the file name “image.png” and it needs to be png format. The mask needs to have the name “mask.png”. 
* Mask properties: axons: 255 ; background and myelin : 0  


## Guideline
* [Guideline to run](https://github.com/neuropoly/axondeepseg/guidelines/guideline.py)


Copyright (c) 2017 NeuroPoly (Polytechnique Montreal)
