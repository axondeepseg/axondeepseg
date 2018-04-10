Introduction
===============================================================================
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin
sheaths from microscopy images. It performs 3-class semantic segmentation using a convolutional neural network.

AxonDeepSeg was developed by NeuroPoly, the neuroimagery laboratory of Polytechnique Montréal.

Installation
===============================================================================
The following lines will help you install all you need to ensure that AxonDeepSeg is working. Test data and
instructions are provided to help you use AxonDeepSeg.

.. NOTE :: AxonDeepSeg is not compatible with Windows due to third-party dependencies.
           AxonDeepSeg was tested with Mac OS and Linux.

Python
-------------------------------------------------------------------------------

First, you should make sure that Python 2.7 is installed on your computer. Run the following command in the terminal::

    python -V

The version of python should be displayed in the terminal. If not, you have to install Python 2.7 on your computer.
To do that, you can follow the instructions given on
`the official python wiki <https://wiki.python.org/moin/BeginnersGuide/Download>`_.

Virtualenv
-------------------------------------------------------------------------------
`Virtualenv` is a Python package that allows you to create virtual environments where
you can sandbox environments with different package versions without affecting
your system packages. If you don't have it installed, please follow the instructions
from the `virtualenv website <https://virtualenv.pypa.io/en/stable/installation/>`_.

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

AxonDeepSeg
-------------------------------------------------------------------------------

Option 1: Installing AxonDeepSeg in application mode (stable release)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. WARNING ::
   Make sure that the virtual environment is activated before you run the following command.

We are now going to install the software AxonDeepSeg.

To install the latest stable release of AxonDeepSeg, you just need to install it with ``pip`` using the following command::

    pip install axondeepseg

.. NOTE ::
   Note that you can install a specific version of the software as follows (replace X.X with the version number, for example 0.2):
   ::
    pip install axondeepseg==X.X

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

Option 2: Installing AxonDeepSeg in development mode (from GitHub)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. WARNING ::
   Make sure that the virtual environment is activated before you run the following command.

To install AxonDeepSeg in development mode, you first need to clone the AxonDeepSeg repository using the following command::

    git clone https://github.com/neuropoly/axondeepseg.git

Then, go to the newly created git repository and install the AxonDeepSeg package using the following commands::

    cd axondeepseg
    pip install -e .

.. NOTE ::
   To update an already cloned AxonDeepSeg package, pull the latest version of the project from GitHub and re-install the application:
   ::
    cd axondeepseg
    git pull
    pip install -e .

The advantage of this installation method over the option 1 (application mode) is that you will always get the last stable version of the package.

Testing the installation
-------------------------------------------------------------------------------

In order to test the installation, you can launch an integrity test by running the following command on the terminal (make sure your virtual env is activated before, as explained in the `Creation a virtual environment <https://neuropoly.github.io/axondeepseg/documentation.html#creating-a-virtual-environment>`_ section)::

    axondeepseg_test


This integrity test automatically performs the axon and myelin segmentation of a test sample. If the test succeeds, the following message will appear in the terminal, meaning that the software was correctly installed::

    * * * Integrity test passed. AxonDeepSeg is correctly installed. * * * 

Existing models
===============================================================================

Two models are available and shipped together with the installation package, so you don't need to install them separately.
The two models are described below:

* A SEM model, that works at a resolution of 0.1 micrometer per pixel.
* A TEM model, that works at a resolution of 0.01 micrometer per pixel.

Getting started
===============================================================================


Example dataset
-------------------------------------------------------------------------------

You can test AxonDeepSeg by downloading the test data available `here <https://osf.io/rtbwc/download>`_. It contains two SEM test samples and one TEM test sample.

Syntax
-------------------------------------------------------------------------------

The script to launch is called **axondeepseg**. It takes several arguments:


**Required arguments:**

-t MODALITY            
                    Type of acquisition to segment.
                    SEM: scanning electron microscopy samples. 
                    TEM: transmission electron microscopy samples.

-i IMGPATH
                    Path to the image to segment or path to the folder where the image(s) to segment is/are located.

**Optional arguments:**

-m MODEL            Folder where the model is located. 
                    The default SEM model path is **default_SEM_model_v1**. 
                    The default TEM model path is **default_TEM_model_v1**.

-s SIZEPIXEL        Pixel size of the image(s) to segment, in micrometers. 
                    If no pixel size is specified, a **pixel_size_in_micrometer.txt** file needs to be added to the image folder path ( that file should contain a single float number corresponding to the resolution of the image, i.e. the pixel size). The pixel size in that file will be used for the segmentation.

-v VERBOSITY        Verbosity level. 
                    **0** (default) : Displays the progress bar for the segmentation. 
                    **1**: Also displays the path of the image(s) being segmented. 
                    **2**: Also displays the information about the prediction step for the segmentation of current sample. 
                    **3**: Also displays the patch number being processed in the current sample.

-o OVERLAP          Overlap value (in pixels) of the patches when doing the segmentation. 
                    Higher values of overlap can improve the segmentation at patch borders, but also increase the segmentation time. Default value: 25. Recommended range of values: [10-100]. 

.. NOTE ::
   You can get the detailed description of all the arguments of the **axondeepseg** command at any time by using the **-h** argument:
   ::
    axondeepseg -h

Segment a single image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To segment a single microscopy image, specify the path to the image to segment in the **-i** argument. For instance, to segment the SEM image **'77.png'** of the test dataset that has a pixel size of 0.07 micrometers, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/77.png -s 0.07

The script will use the explicitely specified size argument (here, 0.07) for the segmentation. If no pixel size is provided in the arguments, it will automatically read the image resolution encoded in the file **'pixel_size_in_micrometer.txt'** if that file exists in the folder containing the image to segment.
The segmented acquisition will be saved in the same folder as the acquisition image, with the suffix **'_seg-axonmyelin.png'**, in *png* format, along with the binary axon and myelin segmentation masks (with the suffixes **'_seg-axon.png'** and **'_seg-myelin.png'**). In our example, the following output files will be generated: **'77_seg-axonmyelin.png'**, **'77_seg-axon.png'** and **'77_seg-myelin.png'**.

To segment the same image by using the **'pixel_size_in_micrometer.txt'** file in the folder (i.e. not specifying the pixel size as argument in the command), use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/77.png

Segment multiple images of the same resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To segment multiple microscopy images of the same resolution that are located in the same folder, specify the path to the folder in the **-i** argument. For instance, to segment the images in folder **'test_sem_image/image 1_sem/'** of the test dataset that have a pixel size of 0.07 micrometers, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image 1_sem/ -s 0.07

To segment multiple images of the same folder and of the same resolution by using the **'pixel_size_in_micrometer.txt'** file in the folder (i.e. not specifying the pixel size as argument in the command), use the following folder structure::

    --folder_with_samples/
    ---- image_1.png
    ---- image_2.png
    ---- image_3.png
    ---- ...
    ---- pixel_size_in_micrometer.txt
    ...

Then, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image 1_sem/

Segment images from multiple folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To segment images that are located in different folders, specify the path to the folders in the **-i** argument, one after the other. For instance, to segment all the images of folders **'test_sem_image/image 1_sem/'** and **'test_sem_image/image 2_sem/'** of the test dataset, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image 1_sem/ test_segmentation/test_sem_image/image 2_sem/

Jupyter notebooks
-------------------------------------------------------------------------------

Here is a list of useful Jupyter notebooks available with AxonDeepSeg:

* `getting_started.ipynb.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/getting_started.ipynb>`_: 
    Notebook that shows how to perform axon and myelin segmentation of a given sample using a Jupyter notebook (i.e. not using the command line tool of AxonDeepSeg). You can also launch this specific notebook without installing and/or cloning the repository by using the `Binder link <https://mybinder.org/v2/gh/neuropoly/axondeepseg/master?filepath=notebooks%2Fgetting_started.ipynb>`_.

* `performance_metrics.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/performance_metrics.ipynb>`_: 
    Notebook that computes a large set of segmentation metrics to assess the axon and myelin segmentation quality of a given sample (compared against a ground truth mask). Metrics include sensitivity, specificity, precision, accuracy, Dice, Jaccard, F1 score, Hausdorff distance.

* `noise_simulation.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/noise_simulation.ipynb>`_: 
    Notebook that simulates various noise/brightness/contrast effects on a given microscopy image in order to assess the robustness of AxonDeepSeg.

* `morphometrics_extraction.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/morphometrics_extraction.ipynb>`_: 
    Notebook that shows how to extract morphometrics from a sample segmented with AxonDeepSeg. The user can extract and save morphometrics for each axon (diameter, solidity, ellipticity, centroid, ...), estimate aggregate morphometrics of the sample from the axon/myelin segmentation (g-ratio, AVF, MVF, myelin thickness, axon density, ...), and generate overlays of axon/myelin segmentation masks, colocoded for axon diameter.

.. NOTE ::
    If it is the first time, install the Jupyter notebook package in the terminal::

        pip install jupyter

    Then, go to the notebooks/ subfolder of AxonDeepSeg and launch a particular notebook as follows::

        cd notebooks
        jupyter notebook name_of_the_notebook.ipynb 


.. WARNING ::
   The current models available for segmentation are trained for patches of 512x512 pixels. This means that your input image(s) should be at least 512x512 pixels in size **after the resampling to the target pixel size of the model you are using to segment**. 

   For instance, the TEM model currently available has a target resolution of 0.01 micrometers per pixel, which means that the minimum size of the input image (in micrometers) is 5.12x5.12.

   **Option:** If your image to segment is too small, you can use padding to artificially increase its size (i.e. add empty pixels around the borders).

Help
===============================================================================

If you experience issues during installation and/or use of AxonDeepSeg, you can post a new issue on the `AxonDeepSeg GitHub issues webpage <https://github.com/neuropoly/axondeepseg/issues>`_. We will reply to you as soon as possible.

Manual correction
-------------------------------------------------------------------------------

If the segmentation with AxonDeepSeg fails or does not give optimal results, you can try one of the following options:

**Option 1: manual correction of the segmentation masks**

* Note that when you launch a segmentation, in the folder output, you will also find the axon and myelin masks (with the suffixes **'_seg-axon.png'** and **'_seg-myelin.png'**). If the segmentation proposed by AxonDeepSeg is not optimal, you can manually correct the myelin mask.
* For the manual correction, we suggest using the GIMP software (https://www.gimp.org/). For a more detailed procedure on how to do the manual correction with GIMP, please consult the following link: `Manual labelling with GIMP <https://docs.google.com/document/d/10E6gzMP6BNGJ_7Y5PkDFmum34U-IcbMi8AvRruhIzvM/edit>`_.

* After correcting the myelin mask, you can regenerate the segmentation masks (axon+myelin). To do this, you can use the following notebook: https://github.com/neuropoly/axondeepseg/blob/master/notebooks/generate_axons_from_myelin.ipynb.

**Option 2: manual correction combined with *AxonSeg* software**

* Manually correct the axon mask (as explained in Option 1).
* Use the `AxonSeg <https://github.com/neuropoly/axonseg>`_ software to segment the myelin from the axon mask. In order to do this, install AxonSeg, and then follow the instructions in part 5 of the `as_tutorial guideline <https://github.com/neuropoly/axonseg/blob/master/as_tutorial.m>`_.

Citation
===============================================================================

If you use this work in your research, please cite:

Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional neural networks. Scientific Reports, 8(1), 3816. `Link to the paper <https://doi.org/10.1038/s41598-018-22181-4>`_.

Licensing
===============================================================================

The MIT License (MIT)

Copyright (c) 2018 NeuroPoly, École Polytechnique, Université de Montréal

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