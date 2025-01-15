.. AxonDeepSeg documentation master file, created by
   sphinx-quickstart on Thu Aug 31 16:00:18 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AxonDeepSeg !
=======================================

Welcome to the AxonDeepSeg framework. In this site you will find the documentation on how to install and how to use AxonDeepSeg to obtain segmentations of your own microscopy data.

AxonDeepSeg is a segmentation software for microscopy data of nerve fibers. It is based on convolutional neural networks. The package also includes a tool for automatic morphometrics computation.

* Graphical user interface: 

.. image:: https://raw.githubusercontent.com/axondeepseg/doc-figures/main/animations/napari.gif
   :alt: Animation of the AxonDeepSeg plugin for Napari showing the segmentation of axon/myelin in a microscopy image.
   :align: center



* AxonDeepSeg pipeline:

.. image:: https://raw.githubusercontent.com/axondeepseg/doc-figures/main/index/fig1.png
   :alt: Figure of AxonDeepSeg's processing pipeline, including the following steps: A) Data Preparation, B) Learning (training of a deep leaning model), C) Evaluation, and D) Prediction on unseen data.
   :align: center

|

* Segmentation of axon/myelin from various species/contrasts:

.. image:: https://raw.githubusercontent.com/axondeepseg/doc-figures/main/index/fig2.png
   :alt: Figure showing detailed sample segmentation results, comparing AxonDeepSeg's predictions of axon/myelin with "Gold standard" ground truth data.
   :align: center

|

* Segmentation of full slice histology and morphometrics extraction:

.. image:: https://raw.githubusercontent.com/axondeepseg/doc-figures/main/index/fig3.png
   :alt: Figure showing a cross section of a rat spinal cord, as well as heatmaps of the spinal cord displaying metrics such as "axon diameter mean", "axon diameter standard deviation", "axon density", "axon volume fraction", "myelin volume fraction", and "g-ratio".
   :align: center

|


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   documentation.rst
   CHANGELOG.md
   license_contributors.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
