Introduction
============
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin
sheaths from microscopy images. It performs 3-class semantic segmentation using a convolutional neural network.

AxonDeepSeg was developed at NeuroPoly Lab, Polytechnique Montreal, University of Montreal, Canada.


Installation
============
The following sections will help you install all the tools you need to run AxonDeepSeg.

.. NOTE :: Starting with Version 2.0, AxonDeepSeg supports the Windows operating system.
           However, please note that our continuous integration testing framework (TravisCI) only tests AxonDeepSeg
           for Unix-style systems, so releases may be more unstable for Windows than Linux/macOS.

Miniconda
---------
Starting with version 3.2.0, AxonDeepSeg is only supported using Python 3.7.x. Although your system may already have
a Python environment installed, we strongly recommend that AxonDeepSeg be used with `Miniconda <https://conda.io/docs/glossary.html#miniconda-glossary>`_, which is a lightweight 
version of the `Anaconda distribution <https://www.anaconda.com/distribution/>`_. Miniconda is typically used to create
virtual Python environments, which provides a separation of installation dependencies between different Python projects. Although
it can be possible to install AxonDeepSeg without Miniconda or virtual environments, we will only provide instructions
for this recommended installation setup.

First, verify if you already have an AxonDeepSeg-compatible version of Miniconda or Anaconda properly installed and is in your systems path. 

In a new terminal window (macOS or Linux) or Anaconda Prompt (Windows – if it is installed), run the following command:::

    conda search python

If a list of available Python versions are displayed and versions >=3.7.0 are available, you may skip to the next section (git).

Linux
~~~~~

To install Miniconda, run the following commands in your terminal:::

    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
    source ~/.bashrc

.. NOTE ::
   If ``conda`` isn't available on new terminal sessions after running these steps, it's possible that your system is configured to use a different startup script. Try adding the line ``source ~/.bashrc`` to your ``~/.bash_profile`` file. `See here <http://www.joshstaiger.org/archives/2005/07/bash_profile_vs.html>`_ for more details.

macOS
~~~~~

To install Miniconda, run the following commands in your terminal:::

    cd ~
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bash_profile
    source ~/.bash_profile

Windows
~~~~~~~

.. NOTE ::
   The AxonDeepSeg installation instruction using the Miniconda have only been tested for Windows 10. Older
   versions of Windows may not be compatible with the tools required to run AxonDeepSeg.

To install Miniconda, go to the `Miniconda installation website <https://conda.io/miniconda.html>`_ and click on the Python 3.x version
installer compatible with your Windows system (64 bit recommended). After the download is complete, execute the
downloaded file, and follow the instructions. If you are unsure about any of the
installation options, we recommend you use the default settings.

git
---
``git`` is a software version control system. Because AxonDeepSeg is hosted on GitHub, a 
service that hosts ``git`` repositories, having ``git`` installed on your system allows you
to download the most up-to-date development version of AxonDeepSeg from a terminal, and 
also allows you to contribute to the project if you wish to do so.

To install ``git``, please follow instructions for your operating system on the 
`git website <https://git-scm.com/downloads>`_

Virtual Environment
-------------------
Virtual environments are a tool to separate the Python environment and packages used 
between Python projects. They allow for different versions of Python packages to be 
installed and managed for the specific needs of your projects. There are several 
virtual environment managers available, but the one we recommend and will use in our installation 
guide is `conda <https://conda.io/docs/>`_, which is installed by default with Miniconda. 
We strongly recommend you create a virtual environment before you continue with your installation.

To create a Python 3.7 virtual environment named "ads_venv", in a terminal window (macOS or Linux) 
or Anaconda Prompt (Windows) run the following command and answer "y" to the installation 
instructions::

    conda create -n ads_venv python=3.7

Then, activate your virtual environment::

    conda activate ads_venv

.. NOTE ::
   To switch back to your default environment, run::

       conda deactivate

AxonDeepSeg
-----------
.. WARNING ::
   Ensure that the virtual environment is activated before you begin your installation.

To install AxonDeepSeg, "clone" AxonDeepSeg's repository (you will need to  
have ``git`` installed on your system)::

    git clone https://github.com/neuropoly/axondeepseg.git

Then, in your Terminal, go to the AxonDeepSeg folder and install the 
AxonDeepSeg package with the following commands::

    cd axondeepseg
    pip install -e .

.. NOTE ::
   To update an already cloned AxonDeepSeg package, pull the latest version of the project from GitHub and re-install the application:
   ::

        cd axondeepseg
        git pull
        pip install -e .

.. WARNING ::
   When re-installing the application, the ``default_SEM_model`` and ``default_TEM_model`` folders in ``AxonDeepSeg/models`` will be deleted and re-downloaded. Please do not store valuable data in these folders.

Testing the installation
------------------------
.. WARNING ::
   Ensure that the virtual environment is activated.

Quick test
~~~~~~~~~~

To test if the software was installed correctly, you can launch a quick integrity test by running the following command on the terminal::

    axondeepseg_test

This integrity test automatically performs the axon and myelin segmentation of a test sample. If the test succeeds, the following message will appear in the terminal::

    * * * Integrity test passed. AxonDeepSeg is correctly installed. * * * 

Comprehensive test
~~~~~~~~~~~~~~~~~~

To run the entire testing suite (more code coverage), go to your AxonDeepSeg project directory on the terminal and run ``py.test``::

    cd axondeepseg
    py.test --cov AxonDeepSeg/ --cov-report term-missing

If all tests pass, AxonDeepSeg was installed succesfully.


Graphical User Interface (GUI) (optional)
-----------------------------------------

AxonDeepSeg can be run via a Graphical User Interface (GUI) instead of the Terminal command line. This GUI is a plugin for the software `FSLeyes <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`_. Beyond the convenience of running AxonDeepSeg with the click of a button, this GUI is also an excellent way to manually correct output segmentations (if need to).

.. image:: _static/GUI_image.png

To install the GUI, you need to install AxonDeepSeg via Github (see instructions above). If you encounter a problem when installing or using the GUI, please report it on our `issue tracker <https://github.com/neuropoly/axondeepseg/issues>`_.
FSLeyes is supported on Mac and Linux. Windows users are encouraged to use a virtual machine if they want to use the GUI.


Once AxonDeepSeg is installed, remain in the virtual environment and follow the OS-specific instructions to install the GUI:


macOS
~~~~~
Install FSLeyes using conda-forge ::

           yes | conda install -c conda-forge fsleyes=0.33.1

Downgrade from latest version of h5py to the most recent working version ::

           yes | conda install -c conda-forge h5py=2.10.0

Launch FSLeyes ::

           fsleyes
           
On the FSLeyes interface, select ``file -> load plugin -> select ads_plugin.py (found in the cloned repository)``
``Install permanently --> yes.``

The plugin is now installed. From now on, you can access the plugin on the FSLeyes interface by selecting ``Settings -> Ortho View -> ADScontrol``.

In case, you find trouble installing FSLeyes plugin for ADS you could refer the video below.

.. raw:: html

   <div style="position: relative; padding-bottom: 5%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
     <iframe width="700" height="394" src="https://www.youtube.com/embed/dz2LqQ5LpIo" frameborder="0" allowfullscreen></iframe>

.. NOTE :: For some users, the ADScontrol tab will not appear after first installing the plugin.
           To resolve this issue, please close FSLeyes and relaunch it (within your virtual environment).
           This step may only be required when you first install the plugin.

Linux (tested on ubuntu)
~~~~~~~~~~~~~~~~~~~~~~~~
Install the C/C++ compilers required to use wxPython ::

           sudo apt-get install build-essential
           sudo apt-get install libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libwebkitgtk-3.0-dev
           sudo apt-get install libjpeg-turbo8-dev libtiff5-dev libsdl1.2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libnotify-dev freeglut3-dev
           
Install wxPython using conda ::

           yes | conda install -c anaconda wxpython
           
Install FSLeyes using conda-forge ::

           yes | conda install -c conda-forge fsleyes=0.33.1

Downgrade from latest version of h5py to the most recent working version ::

           yes | conda install -c conda-forge h5py=2.10.0

Launch FSLeyes ::

           fsleyes

In FSLeyes, do the following:
- Click on ``file -> load plugin``
- Select ``ads_plugin.py`` (found in AxonDeepSeg folder)
- When asked ``Install permanently`` click on ``yes``.

From now on, you can access the plugin on the FSLeyes interface by selecting ``Settings -> Ortho View -> ADScontrol``.

Known issues
~~~~~~~~~~~~
1. The FSLeyes installation doesn't always work on Linux. Refer to the `FSLeyes installation guide <https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/install.html>`_ if you need. In our testing, most issues came from the installation of the wxPython package.


GPU-compatible installation
---------------------------
.. NOTE ::
   This feature is not available if you are using a macOS.

By default, AxonDeepSeg installs the CPU version of TensorFlow. To train a model
using your GPU, you need to uninstall the TensorFlow from your virtual environment, 
and install the GPU version of it::

    pip uninstall tensorflow
    pip install tensorflow-gpu==1.13.1

.. WARNING ::
   Because we recommend the use of version 1.13.1 of Tensorflow GPU, the CUDA version on your system should be 10.0.
   CUDA version less than 10 is not compatible with Tensorflow 1.13.1. To see the CUDA version installed on your system, run ``nvcc --version`` in your Linux terminal.

Existing models
===============

Two models are available and shipped together with the installation package, so you don't need to install them separately.
The two models are described below:

* A SEM model, that works at a resolution of 0.1 micrometer per pixel.
* A TEM model, that works at a resolution of 0.01 micrometer per pixel.

Getting started
===============

Example dataset
---------------

You can test AxonDeepSeg by downloading the test data available `here <https://osf.io/rtbwc/download>`_. It contains two SEM test samples and one TEM test sample.

Syntax
------

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
                    The default SEM model path is **default_SEM_model**. 
                    The default TEM model path is **default_TEM_model**.

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
~~~~~~~~~~~~~~~~~~~~~~

To segment a single microscopy image, specify the path to the image to segment in the **-i** argument. For instance, to segment the SEM image **'77.png'** of the test dataset that has a pixel size of 0.07 micrometers, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/77.png -s 0.07

The script will use the explicitely specified size argument (here, 0.07) for the segmentation. If no pixel size is provided in the arguments, it will automatically read the image resolution encoded in the file **'pixel_size_in_micrometer.txt'** if that file exists in the folder containing the image to segment.
The segmented acquisition will be saved in the same folder as the acquisition image, with the suffix **'_seg-axonmyelin.png'**, in *png* format, along with the binary axon and myelin segmentation masks (with the suffixes **'_seg-axon.png'** and **'_seg-myelin.png'**). In our example, the following output files will be generated: **'77_seg-axonmyelin.png'**, **'77_seg-axon.png'** and **'77_seg-myelin.png'**.

To segment the same image by using the **'pixel_size_in_micrometer.txt'** file in the folder (i.e. not specifying the pixel size as argument in the command), use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/77.png

Segment multiple images of the same resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To segment multiple microscopy images of the same resolution that are located in the same folder, specify the path to the folder in the **-i** argument. For instance, to segment the images in folder **'test_sem_image/image1_sem/'** of the test dataset that have a pixel size of 0.07 micrometers, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/ -s 0.07

To segment multiple images of the same folder and of the same resolution by using the **'pixel_size_in_micrometer.txt'** file in the folder (i.e. not specifying the pixel size as argument in the command), use the following folder structure::

    --folder_with_samples/
    ---- image_1.png
    ---- image_2.png
    ---- image_3.png
    ---- ...
    ---- pixel_size_in_micrometer.txt
    ...

Then, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/

Segment images from multiple folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To segment images that are located in different folders, specify the path to the folders in the **-i** argument, one after the other. For instance, to segment all the images of folders **'test_sem_image/image1_sem/'** and **'test_sem_image/image2_sem/'** of the test dataset, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/ test_segmentation/test_sem_image/image2_sem/

Jupyter notebooks
-----------------

Here is a list of useful Jupyter notebooks available with AxonDeepSeg:

* `getting_started.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/getting_started.ipynb>`_: 
    Notebook that shows how to perform axon and myelin segmentation of a given sample using a Jupyter notebook (i.e. not using the command line tool of AxonDeepSeg). You can also launch this specific notebook without installing and/or cloning the repository by using the `Binder link <https://mybinder.org/v2/gh/neuropoly/axondeepseg/master?filepath=notebooks%2Fgetting_started.ipynb>`_.

* `guide_dataset_building.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/guide_dataset_building.ipynb>`_: 
    Notebook that shows how to prepare a dataset for training. It automatically divides the dataset samples and corresponding label masks in patches of same size.

* `training_guideline.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/training_guideline.ipynb>`_: 
    Notebook that shows how to train a new model on AxonDeepSeg. It also defines the main parameters that are needed in order to build the neural network.

* `performance_metrics.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/performance_metrics.ipynb>`_: 
    Notebook that computes a large set of segmentation metrics to assess the axon and myelin segmentation quality of a given sample (compared against a ground truth mask). Metrics include sensitivity, specificity, precision, accuracy, Dice, Jaccard, F1 score, Hausdorff distance.

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

Guide for manual labelling
==========================

Manual masks for training your own model
----------------------------------------

To be able to train your own model, you will need to manually segment a set of masks. The deep learning model quality will only be as good as your manual masks, so it's important to take care at this step and define your cases.

Technical properties of the manual masks:

* They should be 8-bit PNG files with 1 channel (256 grayscale).
* They should be the same height and width as the images.
* They should contain only 3 unique color values : 0 (black) for background, 127 (gray) for myelin and 255 (white) for axons, and no other intermediate values on strutures edges.
* If you are unfamiliar with those properties, don't worry, the detailed procedures provided in the section below will allow you to follow these guidelines.

Qualitative properties of the manual masks:

* Make sure that every structure (background, myelin or axons) contains only the color of that specific structure (e.g., no black pixels (background) in the axons or the myelin, no white pixels (axons) in the background or myelin, etc.)
* For normal samples without myelin splitting away from the axons, make sure that there is no black pixels (background) on the edges between myelin and axons.

To create a manual mask for training, you can try one of the following:

* Try segmenting your images with AxonDeepSeg's default models and make manual corrections of the segmentation masks in FSLeyes or GIMP software.
* Create a new manual mask using GIMP software.

These options and detailed procedures are described in the section below "Manual correction of segmentation masks".

Here are examples of an image, a good manual mask and a bad manual mask.

.. figure:: _static/image_example.png
    :width: 750px
    :align: center
    :alt: Image example

    Image example

.. figure:: _static/good_mask_example.png
    :width: 750px
    :align: center
    :alt: Good manual mask example

    Good manual mask example

.. figure:: _static/bad_mask_example.png
    :width: 750px
    :align: center
    :alt: Bad manual mask example
    
    Bad manual mask example

Manual correction of segmentation masks
---------------------------------------

If the segmentation with AxonDeepSeg does not give optimal results, you can try one of the following options:

**Option 1: manual correction of the segmentation mask with FSLeyes**

* In FSLeyes, you can make corrections on the myelin segmentation mask using the Edit mode in **Tools > Edit mode**.
* Then, use the **Fill Axons** function to automatically fill the axons and create a corrected axon+myelin mask.
* For a detailed procedure, please consult the following link: `Manual correction with FSLeyes <https://docs.google.com/document/d/1S8i96cJyWZogsMw4RrlQYwglcOWd3HrM5bpTOJE4RBQ/edit>`_.
* As a reference, you can find more informtations about the FSLeyes Edit mode in the `user guide <https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/editing_images.html>`_.

**Option 2: manual labelling with GIMP software**

* To create a new axon+myelin manual mask or to make manual correction on an existing segmentation mask, you can use the GIMP software (`Link for download <https://www.gimp.org/>`_).
* If you are making correction on an existing segmentation mask, note that when you launch a segmentation, in the folder output, you will also find the axon and myelin masks (with the suffixes **'_seg-axon.png'** and **'_seg-myelin.png'**). You can then manually correct the myelin mask and create a corrected axon+myelin mask.
* For a detailed procedure, please consult the following link: `Manual labelling with GIMP <https://docs.google.com/document/d/10E6gzMP6BNGJ_7Y5PkDFmum34U-IcbMi8AvRruhIzvM/edit>`_.

Help
====

If you experience issues during installation and/or use of AxonDeepSeg, you can post a new issue on the `AxonDeepSeg GitHub issues webpage <https://github.com/neuropoly/axondeepseg/issues>`_. We will reply to you as soon as possible.

Citation
========

If you use this work in your research, please cite:

Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional neural networks. Scientific Reports, 8(1), 3816. `Link to the paper <https://doi.org/10.1038/s41598-018-22181-4>`_.

.. include:: ../../CHANGELOG.md

Licensing
=========

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
============

Pierre-Louis Antonsanti, Mathieu Boudreau, Oumayma Bounou, Julien Cohen-Adad, Victor Herman, Melanie Lubrano, Christian Perone, Maxime Wabartha, Aldo Zaimi, Vasudev Sharma, Stoyan Asenov, Marie-Hélène Bourget.
