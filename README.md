# AxonDeepSeg : Axon Deep Segmentation Toolbox
Segment axon and myelin from microscopy data. Written in Python.
Methods based on deep convolutional neural networks.
The U-Net from ([Ronneberger et al.](https://arxiv.org/abs/1505.04597)) is used as a pixel classifier model.
Then an MRF is applied for the post-processing.

The resulting axon segmentation mask feeds a myelin segmentation model ([Zaimi et al.](http://journal.frontiersin.org/article/10.3389/fninf.2016.00037/full))

<img src="https://github.com/neuropoly/axondeepseg/blob/master/doc/schema.jpg" width="600px" align="middle" />

## Installation
### Prerequisites : 
  * **Python 2.7** : https://www.python.org/download/releases/2.7/
  * **PIP** : https://pip.pypa.io/en/stable/installing/
  * **Tensorflow** (GPU enabled, Python 2.7) : https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html
  
### Prerequisites only for myelin segmentation :
  * **Matlab** : https://www.mathworks.com/products/matlab/
  * **AxonSeg (Matlab)** : https://github.com/neuropoly/axonseg

### AxonDeepSeg :

  1. clone repository: ```git clone https://github.com/neuropoly/axondeepseg.git```
  2. cd into axondeepseg, then run: ```python set_config.py -p PATH_AXONSEG -m PATH_MATLAB``` (PATH_AXONSEG contains the AxonSeg Toolbox)
  3. run: ```python setup.py install```

Example dataset:
https://www.dropbox.com/s/juybkrzzafgxkuu/victor.zip?dl=1

##  Data Structure

### To train
<img align="middle" src="https://github.com/neuropoly/axondeepseg/blob/master/doc/stru_train.png" width="300px" />
Mask properties: axons: 255 ; background and myelin : 0  

###  To segment 
<img align="middle" src="https://github.com/neuropoly/axondeepseg/blob/master/doc/stru_seg.png" width="300px"  />

### N.B.
* the image needs to have the file name “image.jpg” and it needs to be jpg format. The mask needs to have the name “mask.jpg”. 
* Mask properties: axons: 255 ; background and myelin : 0  


## Guideline
* [Guideline to run](https://github.com/neuropoly/axondeepseg/blob/master/guideline.py)


Copyright (c) 2016 NeuroPoly (Polytechnique Montreal)
