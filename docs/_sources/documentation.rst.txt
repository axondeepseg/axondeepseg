Introduction
===============================================================================
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin
sheaths from microscopy images.

AxonDeepSeg was developed by NeuroPoly, the neuroimagery laboratory of Polytechnique Montréal.

Getting Started
===============================================================================
The following lines will help you install all you need to ensure that AxonDeepSeg is working. Test data and
instructions are provided to help you use AxonDeepSeg.

.. note:: AxonDeepSeg isn't compatible with Windows due to third-party dependencies.
          AxonDeepSeg was tested with Mac OS and Linux.

Installing python
-------------------------------------------------------------------------------

First, you should make sure that Python 2.7 is installed on your computer. Run the following command in the terminal::

    python -V

The version of python should be displayed in the terminal. If not, you have to install Python 2.7 on your computer.
To do that, you can follow the instructions given on
`the official python wiki <https://wiki.python.org/moin/BeginnersGuide/Download>`_.

Installing virtualenv
-------------------------------------------------------------------------------
`Virtualenv` is a Python package that allows you to create virtual environments where
you can sandbox environments with different package versions without affecting
your system packages. If you don't have it installed, please follow the instructions
from the `virtualenv website <https://virtualenv.pypa.io/en/stable/installation/>`_.


Creating a virtual environment
-------------------------------------------------------------------------------
Before installing AxonDeepSeg, we will need to set up a virtual environment.
A virtual environment is a tool that lets you install specific versions of the python modules you want.
It will allow AxonDeepSeg to run with respect to its module requirements,
without affecting the rest of your python installation.

First, navigate to your home directory::

    cd ~

We will now create a virtual environment. For clarity, we will name it ads_venv::

    virtualenv ads_venv

To activate it, run the following command::

    source ads_venv/bin/activate

If you performed all the steps correctly, your username in the console should now be preceded by the name of your
virtual environment between parenthesis, like this::

    (ads_venv) username@hostname /home/...


Installing AxonDeepSeg
-------------------------------------------------------------------------------
.. WARNING ::
   Make sure that the virtual environment is activated before you run the following command.

We are now going to install the software AxonDeepSeg.

To install AxonDeepSeg, you just need to install it with ``pip`` using the following command::

    pip install axondeepseg

.. note:: If you have an old Mac OS version (<= 10.11), you need to manually install TensorFlow
          dependency by following the instructions from
          `TensorFlow website <https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package>`_.


Models
-------------------------------------------------------------------------------

Two models are available and shipped together with the installation package, so you don't need to install them separately.
The two models are described below:

* A SEM model, that works at a resolution of 0.1 micrometer per pixel.
* A TEM model, that works at a resolution of 0.01 micrometer per pixel.

Data
-------------------------------------------------------------------------------

If you want to test AxonDeepSeg, you can download the test data available
`here <https://osf.io/rtbwc/download>`_.

The images you want to segment must be stored following a particular architecture::

    images_to_segment/
    --folder_1/
    ---- your_image_1.png
    ---- another_image.png
    ---- ...
    ---- pixel_size_in_micrometer.txt (*)
    ...

.. NOTE ::
   The images must be saved in **png format**. You don't have to specifically name them.

* The image file *image.png* is the image to segment.
* The file *pixel_size_in_micrometer.txt* contains a single float number corresponding
to the resolution of the image, that is the **size of a pixel in micrometer**.

.. NOTE ::
   You can also specify the size of a pixel as an argument to our software.



Using AxonDeepSeg
-------------------------------------------------------------------------------

We provide a simple `Jupyter notebook <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/segmentation_image.ipynb>`_ which explains how to use AxonDeepSeg for segmenting axons and myelin. You can directly launch it by using the `Binder link <https://beta.mybinder.org/v2/gh/neuropoly/axondeepseg/master>`_.

To learn to use AxonDeepSeg, you will need some images to segment. If you don't have some,
you can download the test data using the instructions in the `Data <https://neuropoly.github.io/axondeepseg/documentation.html#data>`_ section of this tutorial.



Once you have downloaded the test data, go to the extracted test data folder. In our case::

    cd test_segmentation

The script to launch is called **axondeepseg**. It takes several arguments:

* **t**: type of the image. SEM or TEM.
* **p**: path to the image.
* **s**: (optional) resolution (size in micrometer of a pixel) of the image.
* **v**: (optional) verbosity level. Default 0.

    * 0 displays only a progress bar indicating the advancement of the segmentations.
    * 1 displays additionally the path of the image that was just segmented.
    * 2 displays additionally information about the current step of the segmentation of the current image.

To segment one of the image that we downloaded (here, a SEM image), run the following command::

    axondeepseg -t SEM -p test_sem_image/image1_sem/77.png -v 2 -s 0.07

The script will use the size argument (here, 0.07) for the segmentation. If no size is provided in the arguments,
it will automatically read the image resolution encoded in the file: *pixel_size_in_micrometer.txt*
The different steps will be displayed in the terminal thanks to the verbosity level set to 2.
The segmented acquisition itself will be saved in the same folder as the acquisition image,
with the prefix 'segmentation_', in png format.


* To segment multiple images acquired with the same resolution, put them all in the same folder and
launch the segmentation of this folder, like below::

    axondeepseg -t SEM -p test_sem_image/image 1_sem/


* To segment multiple images acquired with different resolutions,
please use the folder structure explained in https://neuropoly.github.io/axondeepseg/documentation.html#data,
i.e., put all image with the same resolution in the same folder.
* Then, segment each folder one after the other using the argument **-s** or segment all folders in one command
by specifying multiple paths to segment and using a different pixel_size_in_micrometer.txt for each folder, like this::

    axondeepseg -t SEM -p test_sem_image/image1_sem test_sem_image/image2_sem/


Here, we segment all images located in image1_sem and image2_sem that don't have the "segmented" suffix.

Each output segmentation will be saved in the corresponding sub-folder.

Citation
===============================================================================

If you use this work in your research, please cite:

TODO add arxiv link

Licensing
===============================================================================

The MIT License (MIT)

Copyright (c) 2017 NeuroPoly, École Polytechnique, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contributors
===============================================================================

Pierre-Louis Antonsanti, Julien Cohen-Adad, Victor Herman, Christian Perone, Maxime Wabartha, Aldo Zaimi