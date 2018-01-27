Introduction
===============================================================================
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin
sheaths from microscopy images.

AxonDeepSeg was developed by NeuroPoly, the neuroimagery laboratory of Polytechnique Montréal.

Getting Started
===============================================================================
The following lines will help you install all you need to ensure that AxonDeepSeg is working. Test data and
instructions are provided to help you use AxonDeepSeg.

.. note:: AxonDeepSeg is not compatible with Windows due to third-party dependencies.
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

.. WARNING ::    
  If you experience the following error:
  "Could not find a version that satisfies the requirement tensorflow>=XXX (from axondeepseg) (from versions: )... ",
  you will need to manually install the TensorFlow dependency.

  For OS X users, run the following command to install TensorFlow 1.3.0:
  :: 
    pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py2-none-any.whl

  For Linux users, run the following command to install TensorFlow 1.3.0:
  ::
    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp34-cp34m-linux_x86_64.whl

  You can get more information by following the instructions from the `TensorFlow website <https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package>`_.

  **Once TensorFlow is installed, re-run the pip command:**
  :: 
    pip install axondeepseg


.. NOTE ::
   In order to test the installation, you can launch an integrity test by running the following command on the terminal (make sure your virtual env is activated before, as explained in the `Creation a virtual environment <https://neuropoly.github.io/axondeepseg/documentation.html#creating-a-virtual-environment>`_ section) :
   ::
    axondeepseg_test

   This integrity test automatically performs the axon and myelin segmentation of a test sample. If the test succeeds, the following message will appear in the terminal, meaning that the  software was correctly installed:
   ::
    * * * Integrity test passed. AxonDeepSeg is correctly installed. * * * 


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
   You can also specify the pixel size as an argument to our software (see next section).



Using AxonDeepSeg
-------------------------------------------------------------------------------

We provide a simple `Jupyter notebook <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/getting_started.ipynb>`_ which explains how to use AxonDeepSeg for segmenting axons and myelin. You can directly launch it by using the `Binder link <https://mybinder.org/v2/gh/neuropoly/axondeepseg/master?filepath=notebooks%2Fgetting_started.ipynb>`_.

To learn to use AxonDeepSeg, you will need some images to segment. If you don't have some,
you can download the test data using the instructions in the `Data <https://neuropoly.github.io/axondeepseg/documentation.html#data>`_ section of this tutorial.

.. WARNING ::
   The current models available for segmentation are trained for patches of 512x512 pixels. This means that your input image(s) should be at least 512x512 pixels in size **after the resampling to the target pixel size of the model you are using to segment**. 

   For instance, the TEM model currently available has a target resolution of 0.01 micrometers per pixel, which means that the minimum size of the input image (in micrometers) is 5.12x5.12.

   **Option:** If your image to segment is too small, you can use padding to artificially increase its size (i.e. add empty pixels around the borders).


Once you have downloaded the test data, go to the extracted test data folder. In our case::

    cd test_segmentation

The script to launch is called **axondeepseg**. It takes several arguments:

* **t**: type of the image. SEM or TEM.
* **i**: path to the image.
* **s**: (optional) resolution (size in micrometer of a pixel) of the image.
* **v**: (optional) verbosity level. Default 0.

    * 0 displays only a progress bar indicating the advancement of the segmentations.
    * 1 displays additionally the path of the image that was just segmented.
    * 2 displays additionally information about the current step of the segmentation of the current image.

To segment one of the image that we downloaded (here, a SEM image), run the following command::

    axondeepseg -t SEM -i test_sem_image/image1_sem/77.png -v 2 -s 0.07

The script will use the size argument (here, 0.07) for the segmentation. If no size is provided in the arguments,
it will automatically read the image resolution encoded in the file: *pixel_size_in_micrometer.txt*
The different steps will be displayed in the terminal thanks to the verbosity level set to 2.
The segmented acquisition itself will be saved in the same folder as the acquisition image,
with the suffix 'segmented_', in png format.


* To segment multiple images acquired with the same resolution, put them all in the same folder and
launch the segmentation of this folder, like below::

    axondeepseg -t SEM -i test_sem_image/image 1_sem/


* To segment multiple images acquired with different resolutions,
please use the folder structure explained in `Data <https://neuropoly.github.io/axondeepseg/documentation.html#data>`_,
i.e., put all image with the same resolution in the same folder.
* Then, segment each folder one after the other using the argument **-s** or segment all folders in one command
by specifying multiple paths to segment and using a different pixel_size_in_micrometer.txt for each folder, like this::

    axondeepseg -t SEM -i test_sem_image/image1_sem test_sem_image/image2_sem/


Here, we segment all images located in image1_sem and image2_sem that don't have the "segmented" suffix.

Each output segmentation will be saved in the corresponding sub-folder.


Post-processing tools
-------------------------------------------------------------------------------

If the segmentation with AxonDeepSeg fails or does not give optimal results, you can try one of the following options:

**Option 1: manual correction of the segmentation masks**

* Note that when you launch a segmentation, in the folder output, you will also find the axon and myelin masks (separately), named 'axon_mask.png' and 'myelin_mask.png'. If the segmentation proposed by AxonDeepSeg is not optimal, you can manually correct the myelin mask.
* For example, you can open the microscopy image and the myelin mask with an external tool/software (such as GIMP: https://www.gimp.org/). For a more detailed procedure, you can visit https://www.gimp.org/tutorials/Layer_Masks/.
* After correcting the myelin mask, you can regenerate the image (axon+myelin). To do this, you can use the following notebook: https://github.com/neuropoly/axondeepseg/blob/master/notebooks/generate_axons_from_myelin.ipynb.

**Option 2: manual correction combined with AxonSeg software**

* Manually correct the axon mask (as explained in Option 1).
* Use the `AxonSeg <https://github.com/neuropoly/axonseg>`_ software to segment the myelin from the axon mask. In order to do this, install AxonSeg, and then follow the instructions in part 5 of the `as_tutorial guideline <https://github.com/neuropoly/axonseg/blob/master/as_tutorial.m>`_.


Help
===============================================================================

If you experience issues during installation and/or use of AxonDeepSeg, you can post a new issue on the `AxonDeepSeg GitHub issues webpage <https://github.com/neuropoly/axondeepseg/issues>`_. We will reply to you as soon as possible.


Citation
===============================================================================

If you use this work in your research, please cite:

Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2017). AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional neural networks. arXiv Preprint arXiv:1711.01004. `Link to the paper <https://arxiv.org/abs/1711.01004>`_.

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