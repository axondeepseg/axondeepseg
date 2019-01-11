Introduction
===============================================================================
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin
sheaths from microscopy images. It performs 3-class semantic segmentation using a convolutional neural network.

AxonDeepSeg was developed by NeuroPoly, the neuroimagery laboratory of Polytechnique Montréal.

Changelog
===============================================================================

Version [2.2dev] - XXXX-XX-XX
-------------------------------------------------------------------------------

**Changed:**

- Resolve image rescale warnings
- Handle exception for images smaller than minimum patch size after resizing
- Revert tensorflow requirekment to 1.3.0 and remove tifffile requirement

Version [2.1] - 2018-09-25
-------------------------------------------------------------------------------

**Changed:**

- Fixed bug that would crash when user inputed consent for Sentry tracking

Version [2.0] - 2018-09-11
-------------------------------------------------------------------------------

**Changed:**

- Upgraded ADS for Python 3.6-compatibility (no longer supporting Python 2.7)
- Minor changes to make ADS Windows-compatibile
- Removed plot hold commands (deprecated)

Version [1.1] - 2018-08-02
-------------------------------------------------------------------------------

**Changed:**

- Minor Mac OSX-related bug fix
- Changed installation requirements to exact release versions

Version [1.0] - 2018-08-02
-------------------------------------------------------------------------------

Versions 1.x will remain Python 2.7-compatible

Version [0.6] - 2018-08-01
-------------------------------------------------------------------------------

(version 0.5 was skipped due to conflicting file on PyPI)

**Added:**

- Comprehensive testing suite
- Bug tracking (Sentry)
- Blue-red visualisation function for segmented masks

**Changed:**

- Dataset building and training notebook
- Minor documentation improvements
- Minor bug fixes

Version [0.4.1] - 2018-05-16
-------------------------------------------------------------------------------

**Added:**

- GIMP procedure for ground truth labelling or segmentation correction added in the documentation.
- Compatibility with tiff images.
- Continuous integration with Travis is now supported.

**Changed:**

- The documentation website is now hosted on ReadTheDocs.
- Updated documentation on the usage of AxonDeepSeg.
- Change of axon and myelin masks filenames for better clarity.

Version [0.3] - 2018-02-22
-------------------------------------------------------------------------------

**Added:**

- Compatibility for image inputs other than png
- Pre-processing of input images is now done inside AxonDeepSeg

**Changed:**

- Help display when running AxonDeepSeg from terminal

Installation
===============================================================================
The following sections will help you install all the tools you need to run AxonDeepSeg.

.. NOTE :: Starting with Version 2.0, AxonDeepSeg supports the Windows operating system.
           However, please note that our continuous integration testing framework (TravisCI) only tests AxonDeepSeg
           for Unix-style systems, so releases may be more unstable for Windows than Linux/macOS.

Miniconda
-------------------------------------------------------------------------------
Starting with versions 2.0+, AxonDeepSeg is only supported using Python 3.0. Although your system may already have
a Python environment installed, we strongly recommend that AxonDeepSeg be used with `Miniconda <https://conda.io/docs/glossary.html#miniconda-glossary>`_, which is a lightweight version
version of the `Anaconda distribution <https://www.anaconda.com/distribution/>`_. Miniconda is typically used to create
virtual Python environments, which provides a separation of installation dependencies between different Python projects. Although
it can be possible to install AxonDeepSeg without Miniconda or virtual environments, we will only provide instructions
for this recommended installation setup.

First, verify if you already have an AxonDeepSeg-compatible version of Miniconda or Anaconda properly installed and is in your systems path. 

In a new terminal window (macOS or Linux) or Anaconda Prompt (Windows – if it is installed), run the following command:::

    conda search python

If a list of available Python versions are displayed and versions >=3.6.0 are available, you may skip to the next section (Git).

Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install Miniconda, run the following commands in your terminal:::

    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
    source ~/.bashrc

macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install Miniconda, run the following commands in your terminal:::

    cd ~
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bash_profile
    source ~/.bash_profile

Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install Miniconda, go to the `Miniconda installation website <https://conda.io/miniconda.html>`_ and click on the Python 3.x version
installer compatible with your Windows system (64 bit recommended). After the download is complete, execute the
downloaded file, and follow the instructions. If you are unsure about any of the
installation options, we recommend you use the default settings.

Git (optional)
-------------------------------------------------------------------------------
Git is a software version control system. Because AxonDeepSeg is hosted on GitHub, a 
service that hosts Git repositories, having Git installed on your system allows you
to download the most up-to-date development version of AxonDeepSeg from a terminal, and 
also allows you to contribute to the project if you wish to do so.

Although an optional step (AxonDeepSeg can also be downloaded other ways, see below), if you 
want to install Git, please follow instructions for your operating system on the 
`Git website <https://git-scm.com/downloads>`_

Virtual Environment
-------------------------------------------------------------------------------
Virtual environments are a tool to separate the Python environment and packages used 
between Python projects. They allow for different versions of Python packages to be 
installed and managed for the specific needs of your projects. There are several 
virtual environment managers available, but the one we recommend and will use in our installation 
guide is `conda <https://conda.io/docs/>`_, which is installed by default with Miniconda. 
We strongly recommend you create a virtual environment before you continue with your installation.

To create a Python 3.6 virtual environment named "ads_venv", in a terminal window (macOS or Linux) 
or Anaconda Prompt (Windows) run the following command and answer "y" to the installation 
instructions::

    conda create -n ads_venv python=3.6

Then, activate your virtual environment::

    conda activate ads_venv

.. NOTE ::
   To switch back to your default environment, run::

       conda deactivate

AxonDeepSeg
-------------------------------------------------------------------------------
.. WARNING ::
   Ensure that the virtual environment is activated before you begin your installation.

Latest version (development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the latest version of AxonDeepSeg (development), we recommend that you clone the AxonDeepSeg repository 
if you have ``git`` installed on your system::

    git clone https://github.com/neuropoly/axondeepseg.git

Otherwise, download and extract AxonDeepSeg by clicking `this link <https://github.com/neuropoly/axondeepseg/archive/master.zip>`_.

Then, in your terminal window, go to the AxonDeepSeg folder and install the 
AxonDeepSeg package. The following ``cd`` command assumes that you followed the ``git clone``
instruction above::

    cd axondeepseg
    pip install -e .

.. NOTE ::
   If you downloaded AxonDeepSeg using the link above instead of ``git clone``, you may need to ``cd`` to a different folder (e.g. ``Downloads`` folder 
   located within your home folder ``~``), and the AxonDeepSeg folder may have a different name (e.g. ``axondeepseg-master``).

.. NOTE ::
   To update an already cloned AxonDeepSeg package, pull the latest version of the project from GitHub and re-install the application:
   ::

        cd axondeepseg
        git pull
        pip install -e .

Stable release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can install the latest stable release of AxonDeepSeg using ``pip`` with the following command::

    pip install axondeepseg

Testing the installation
-------------------------------------------------------------------------------
.. WARNING ::
   Ensure that the virtual environment is activated.

Quick test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test if the software was installed correctly, you can launch a quick integrity test by running the following command on the terminal::

    axondeepseg_test

This integrity test automatically performs the axon and myelin segmentation of a test sample. If the test succeeds, the following message will appear in the terminal::

    * * * Integrity test passed. AxonDeepSeg is correctly installed. * * * 

Comprehensive test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. NOTE ::
   This feature is not available if you installed AxonDeepSeg using ``pip``.

To run the entire testing suite (more code coverage), go to your AxonDeepSeg project directory on the terminal and run ``py.test``::

    cd axondeepseg
    py.test --cov AxonDeepSeg/ --cov-report term-missing

If all tests pass, AxonDeepSeg was installed succesfully.

GPU-compatible installation
--------------------------------------------------------------------------------
.. NOTE ::
   This feature is not available if you installed AxonDeepSeg using ``pip``,
   or if you are using a macOS.

By default, AxonDeepSeg installs the CPU version of TensorFlow. To train a model
using your GPU, you need to uninstall the TensorFlow from your virtual environment, 
and install the GPU version of it::

    pip uninstall tensorflow
    pip install tensorflow-gpu==1.3.0

.. WARNING ::
   Because we recommend the use of version 1.3.0 of Tensorflow GPU, the CUDA installation on your system should be 8.0.
   CUDA 9.0+ is not compatible with Tensorflow 1.3.0. To see the CUDA version installed on your system, run ``nvcc --version`` in your Linux terminal.

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

* `getting_started.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/getting_started.ipynb>`_: 
    Notebook that shows how to perform axon and myelin segmentation of a given sample using a Jupyter notebook (i.e. not using the command line tool of AxonDeepSeg). You can also launch this specific notebook without installing and/or cloning the repository by using the `Binder link <https://mybinder.org/v2/gh/neuropoly/axondeepseg/master?filepath=notebooks%2Fgetting_started.ipynb>`_.

* `guide_dataset_building.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/guide_dataset_building.ipynb>`_: 
    Notebook that shows how to prepare a dataset for training. It automatically divides the dataset samples and corresponding label masks in patches of same size.

* `training_guideline.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/training_guideline.ipynb>`_: 
    Notebook that shows how to train a new model on AxonDeepSeg. It also defines the main parameters that are needed in order to build the neural network.

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

Pierre-Louis Antonsanti, Mathieu Boudreau, Oumayma Bounou, Julien Cohen-Adad, Victor Herman, Melanie Lubrano, Christian Perone, Maxime Wabartha, Aldo Zaimi.
