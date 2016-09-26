# AxonDeepSeg

Axon segmentation with microscopic images and based on deep convolutional neural networks.
The U-Net from ([ Ronneberger et al.](https://arxiv.org/abs/1505.04597)) is used as a pixel classifier model.
Then an MRF is applied for the post-processing.
The resulting axon segmentation mask feeds a myelin detector ([Zaimi et al.](http://journal.frontiersin.org/article/10.3389/fninf.2016.00037/full))

<img src="https://github.com/vherman3/AxonSegmentation/blob/master/schema.png" width="600px" align="middle" />

# Guideline #
* [Guideline presentation](https://docs.google.com/presentation/d/1gtp8UiqJJF7pRaBctTryoGQPACMeu29DhCMYn6k6PXQ/edit?usp=sharing)
* [Guideline to run](https://github.com/vherman3/AxonSegmentation/blob/master/guideline.py)
