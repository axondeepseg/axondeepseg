Introduction
============
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin sheaths from microscopy images. It performs 3-class semantic segmentation using a convolutional neural network.

AxonDeepSeg was developed at NeuroPoly Lab, Polytechnique Montreal, University of Montreal, Canada.


Installation
============
The following sections will help you install all the tools you need to run AxonDeepSeg.

.. NOTE :: Starting with Version 2.0, AxonDeepSeg supports the Windows operating system. However, please note that our continuous integration testing framework (TravisCI) only tests AxonDeepSeg for Unix-style systems, so releases may be more unstable for Windows than Linux/macOS.

Miniconda
---------
Starting with version 4.0.0, AxonDeepSeg is only supported using Python 3.8.x. Although your system may already have a Python environment installed, we strongly recommend that AxonDeepSeg be used with `Miniconda <https://conda.io/docs/glossary.html#miniconda-glossary>`_, which is a lightweight version of the `Anaconda distribution <https://www.anaconda.com/distribution/>`_. Miniconda is typically used to create virtual Python environments, which provides a separation of installation dependencies between different Python projects. Although it can be possible to install AxonDeepSeg without Miniconda or virtual environments, we will only provide instructions for this recommended installation setup.

First, verify if you already have an AxonDeepSeg-compatible version of Miniconda or Anaconda properly installed and is in your systems path. 

In a new terminal window (macOS or Linux) or Anaconda Prompt (Windows – if it is installed), run the following command:::

    conda search python

If a list of available Python versions are displayed and versions >=3.8.0 are available, you may skip to the next section (git).

Linux
~~~~~

To install Miniconda, run the following commands in your terminal:::

    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
    source ~/.bashrc

.. NOTE :: If ``conda`` isn't available on new terminal sessions after running these steps, it's possible that your system is configured to use a different startup script. Try adding the line ``source ~/.bashrc`` to your ``~/.bash_profile`` file. `See here <http://www.joshstaiger.org/archives/2005/07/bash_profile_vs.html>`_ for more details.

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

.. NOTE :: The AxonDeepSeg installation instruction using the Miniconda have only been tested for Windows 10. Older versions of Windows may not be compatible with the tools required to run AxonDeepSeg.

To install Miniconda, go to the `Miniconda installation website <https://conda.io/miniconda.html>`_ and click on the Python 3.x version installer compatible with your Windows system (64 bit recommended). After the download is complete, execute the downloaded file, and follow the instructions. If you are unsure about any of the installation options, we recommend you use the default settings.

git
---
``git`` is a software version control system. Because AxonDeepSeg is hosted on GitHub, a service that hosts ``git`` repositories, having ``git`` installed on your system allows you to download the most up-to-date development version of AxonDeepSeg from a terminal, and also allows you to contribute to the project if you wish to do so.

To install ``git``, please follow instructions for your operating system on the `git website <https://git-scm.com/downloads>`_

Install AxonDeepSeg
-------------------

To install AxonDeepSeg, "clone" AxonDeepSeg's repository (you will need to have ``git`` installed on your system) and then open the directory::

    git clone https://github.com/neuropoly/axondeepseg.git
    cd axondeepseg

Virtual environments are a tool to separate the Python environment and packages used between Python projects. They allow for different versions of Python packages to be installed and managed for the specific needs of your projects. There are several virtual environment managers available, but the one we recommend and will use in our installation guide is `conda <https://conda.io/docs/>`_, which is installed by default with Miniconda. We strongly recommend you create a virtual environment before you continue with your installation.

To setup a Python virtual environment named "ads_venv" with all the required packages, in a terminal window (macOS or Linux) or Anaconda Prompt (Windows) run the following command and answer "y" to the installation instructions::

    conda env create -f environment.yml -n ads_venv

.. WARNING :: For some users, the installation may take up to 30 minutes as many dependencies have shared subdependencies, and resolving these potential conflicts takes time. If that's the case, we encourage you to take a break from your screen and go for a walk while listening to the `AxonDeepSeg Spotify playlist <https://open.spotify.com/playlist/27LVNnfhTKjVOli6bPCaV5?si=OydcwxoOSamwCsg3xcqybw>`_.

.. NOTE :: FSLeyes is only supported on Mac and Linux. Windows users are encouraged to use a virtual machine if they want to use the GUI. 

Then, activate your virtual environment::

    conda activate ads_venv

.. NOTE :: To switch back to your default environment, run:
  ::

       conda deactivate

Once your virtual environment is installed and activated, install the AxonDeepSeg software with the following commands::

    pip install -e .

.. NOTE :: To update an already cloned AxonDeepSeg package, pull the latest version of the project from GitHub and re-install the application:
   ::

        cd axondeepseg
        git pull
        pip install -e .

.. WARNING :: When re-installing the application, the model folders in ``AxonDeepSeg/models`` will be deleted and re-downloaded. Please do not store valuable data in these folders.

.. raw:: html

   <iframe width="700" height="394" src="https://www.youtube.com/embed/7RwZihR0HWk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



Testing the installation
------------------------
.. WARNING :: Ensure that the virtual environment is activated.

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


Graphical User Interface (GUI)
-----------------------------------------

AxonDeepSeg can be run via a Graphical User Interface (GUI) instead of the Terminal command line. This GUI is a plugin for the software `FSLeyes <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`_. Beyond the convenience of running AxonDeepSeg with the click of a button, this GUI is also an excellent way to manually correct output segmentations (if need to).

.. image:: https://raw.githubusercontent.com/axondeepseg/doc-figures/main/introduction/GUI_image.png

Launch FSLeyes ::

           fsleyes

In FSLeyes, do the following:
- Click on ``file -> load plugin``
- Select ``ads_plugin.py`` (found in AxonDeepSeg folder)
- When asked ``Install permanently`` click on ``yes``.
- Close FSLeyes and re-open it.

From now on, you can access the plugin on the FSLeyes interface by selecting ``Settings -> Ortho View -> ADScontrol``.

In case, you find trouble installing FSLeyes plugin for ADS you could refer the video below.

.. raw:: html

   <iframe width="700" height="394" src="https://www.youtube.com/embed/ImElcp9_k6Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


.. NOTE :: For some users, the ADScontrol tab will not appear after first installing the plugin.
           To resolve this issue, please close FSLeyes and relaunch it (within your virtual environment).
           This step may only be required when you first install the plugin.


Known issues
~~~~~~~~~~~~
1. The FSLeyes installation doesn't always work on Linux. Refer to the `FSLeyes installation guide <https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/install.html>`_ if you need. In our testing, most issues came from the installation of the wxPython package.

GPU Support
-----------

If you have a compatible NVIDIA GPU card that supports CUDA11 and with the right driver installed, running the following command after installing AxonDeepSeg will install the necessary ``torch`` and ``torchvision`` versions for GPU use of AxonDeepSeg::

    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 --find-links https://download.pytorch.org/whl/torch_stable.html

For more information aboud CPU and GPU support, please view the `IVADOMED documentation page <https://ivadomed.org/installation.html#step-3-install-torch-and-torchvision-with-cpu-or-gpu-support>`_ about it.

Existing models
===============

Three models are available and shipped together with the installation package, so you don't need to install them separately.
The three models are described below:

* A SEM model, that works at a resolution of 0.1 micrometer per pixel.
* A TEM model, that works at a resolution of 0.01 micrometer per pixel.
* A BF (bright-field) model, that works at a resolution of 0.1 micrometer per pixel.

Using AxonDeepSeg
=================

Activate the virtual environment
--------------------------------

To use AxonDeepSeg, you must first activate the virtual environment if it isn't currently activated. To do so, run::

    conda activate ads_venv

.. NOTE :: To switch back to your default environment, run:
  ::

       conda deactivate

Example dataset
---------------

You can demo the AxonDeepSeg by downloading the test data available `here <https://api.github.com/repos/axondeepseg/data-example/zipball>`_. It contains two SEM test samples and one TEM test sample.

Segmentation
------------

Syntax
~~~~~~

The script to launch is called **axondeepseg**. It takes several arguments:


**Required arguments:**

-t MODALITY            
                    Type of acquisition to segment.
                    SEM: scanning electron microscopy samples. 
                    TEM: transmission electron microscopy samples.
                    BF: bright field optical microscopy samples.

-i IMGPATH
                    Path to the image to segment or path to the folder where the image(s) to segment is/are located.

**Optional arguments:**

-m MODEL            Folder where the model is located, if different from the default model.

-s SIZEPIXEL        Pixel size of the image(s) to segment, in micrometers. 
                    If no pixel size is specified, a **pixel_size_in_micrometer.txt** file needs to be added to the image folder path ( that file should contain a single float number corresponding to the resolution of the image, i.e. the pixel size). The pixel size in that file will be used for the segmentation.

-v VERBOSITY        Verbosity level. 
                    **0** (default): Quiet mode. Shows minimal information on the terminal.
                    **1**: Developer mode. Shows more information on the terminal, useful for debugging.. 

--overlap           Overlap value (in pixels) of the patches when doing the segmentation. 
                    Higher values of overlap can improve the segmentation at patch borders, but also increase the segmentation time. Default value: 48. Recommended range of values: [10-100]. 

-z ZOOM             Zoom factor.
                    When applying the model, the pixel size of the image will be multiplied by this number.

.. NOTE :: You can get the detailed description of all the arguments of the **axondeepseg** command at any time by using the **-h** argument:
   ::

        axondeepseg -h

Segment a single image
^^^^^^^^^^^^^^^^^^^^^^

To segment a single microscopy image, specify the path to the image to segment in the **-i** argument. For instance, to segment the SEM image **'77.png'** of the test dataset that has a pixel size of 0.07 micrometers, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/77.png -s 0.07

The script will use the explicitely specified size argument (here, 0.07) for the segmentation. If no pixel size is provided in the arguments, it will automatically read the image resolution encoded in the file **'pixel_size_in_micrometer.txt'** if that file exists in the folder containing the image to segment.
The segmented acquisition will be saved in the same folder as the acquisition image, with the suffix **'_seg-axonmyelin.png'**, in *png* format, along with the binary axon and myelin segmentation masks (with the suffixes **'_seg-axon.png'** and **'_seg-myelin.png'**). In our example, the following output files will be generated: **'77_seg-axonmyelin.png'**, **'77_seg-axon.png'** and **'77_seg-myelin.png'**.

To segment the same image by using the **'pixel_size_in_micrometer.txt'** file in the folder (i.e. not specifying the pixel size as argument in the command), use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/77.png

Segment multiple images of the same resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To segment images that are located in different folders, specify the path to the folders in the **-i** argument, one after the other. For instance, to segment all the images of folders **'test_sem_image/image1_sem/'** and **'test_sem_image/image2_sem/'** of the test dataset, use the following command::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/ test_segmentation/test_sem_image/image2_sem/

Segment images using a zoom factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, the quality of the segmentation can be improved by tweaking the pixel size so that your data's resampled resolution, for example, better matches the model's input resolution. 
This is why we provide the **-z** argument, which lets you specify a zoom factor to adjust your pixel size. Note that this option also works for multiple images or multiple folders. 

For example, a pixel size of 0.07 with a zoom factor of 2.0 will provide a pixel size of 0.14 to the model before running inference::

    axondeepseg -t SEM -i test_segmentation/test_sem_image/image1_sem/77.png -s 0.07 -z 2.0

Morphometrics
-------------

You can generate morphometrics using AxonDeepSeg via the command line interface.

Syntax
~~~~~~

The script to launch is called **axondeepseg_morphometrics**. It has several arguments.

**Required arguments:**

-i IMGPATH
                    Path to the image file whose morphometrics needs to be calculated.

**Optional arguments:**

-s SIZEPIXEL        Pixel size of the image(s) to segment, in micrometers. 
                    If no pixel size is specified, a **pixel_size_in_micrometer.txt** file needs to be added to the image folder path (that file should contain a single float number corresponding to the resolution of the image, i.e. the pixel size). The pixel size in that file will be used for the morphometrics computation.

-a AXONSHAPE       Axon shape
                    **circle:** Axon shape is considered as circle. In this case, diameter is computed using equivalent diameter. 
                    **ellipse:** Axon shape is considered as an ellipse. In this case, diameter is computed using ellipse minor axis.
                    The default axon shape is set to **circle**.

-f FILENAME         Name of the excel file in which the morphometrics will be stored.
                    The excel file extension can either be **.xlsx** or **.csv**.
                    If name of the excel file is not provided, the morphometrics will be saved as **axon_morphometrics.xlsx**.

Morphometrics of a single image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Before computing the morphometrics of an image, make sure it has been segmented using AxonDeepSeg ::

    axondeepseg_morphometrics -i test_segmentation/test_sem_image/image1_sem/77.png -a circle -f axon_morphometrics 

This generates a **'77_axon_morphometrics.xlsx'** file in the image directory::

    --image1_sem/
    ---- 77.png
    ---- 77_seg-axon.png
    ---- 77_seg-axonmyelin.png
    ---- 77_seg-myelin.png
    ---- 77_axon_morphometrics.xlsx
    ---- pixel_size_in_micrometer.txt

    ...

.. NOTE 1:: If name of the excel file is not provided using the `-f` flag of the CLI, the morphometrics will be saved as the original image name with suffix "axon_morphometrics.xlsx". However, if custom filename is provided, then the morphometrics will be saved as the original image name with suffix "custom filename".
   ::
.. NOTE 2:: By default, AxonDeepSeg treats axon shape as **circle** and the calculation of the diameter is based on the axon area of the mask. 
           For each axons, the equivalent diameter is computed, which is the diameter of a circle with the same area as the axon. ::
           
           If you wish to treat axon shape as an ellipse, you can set the  **-a** argument to be **ellipse**.
           When axon shape is set to ellipse, the calculation of the diameter is based on ellipse minor axis::
            
            axondeepseg -i test_segmentation/test_sem_image/image1_sem/77.png -a ellipse

Morphometrics of a specific image from multiple folders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To generate morphometrics of images which are located in different folders, specify the path of the image folders using the **-i** argument of the CLI separated by space. For instance, to compute morphometrics of the image **'77.png'** and **'image.png'** present in the folders **'test_sem_image/image1_sem/'** and **'test_sem_image/image2_sem/'** respectively of the test dataset, use the following command::

    axondeepseg_morphometrics -i test_segmentation/test_sem_image/image1_sem/77.png test_segmentation/test_sem_image/image2_sem/image.png

This will generate **'77_axon_morphometrics.xlsx'** and **'image_axon_morphometrics.xlsx'** files in the **'image1_sem'** and **'image2_sem'** folders:: 

    --image1_sem/
    ---- 77.png
    ---- 77_seg-axon.png
    ---- 77_seg-axonmyelin.png
    ---- 77_seg-myelin.png
    ---- 77_axon_morphometrics.xlsx
    ---- pixel_size_in_micrometer.txt

    ...

    --image2_sem/
    ---- image.png
    ---- image_seg-axon.png
    ---- image_seg-axonmyelin.png
    ---- image_seg-myelin.png
    ---- image_axon_morphometrics.xlsx
    ---- pixel_size_in_micrometer.txt

Morphometrics of all the images present in folder(s)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To compute the morphometrics of batches of images present in folder(s), input the path of the directories using the **-i** argument separated by space. For example, the morphometrics files of the images present in the directories  **'test_sem_image/image1_sem/'** and **'test_sem_image/image2_sem/'** are computed using the following CLI command::

    axondeepseg_morphometrics -i test_segmentation/test_sem_image/image1_sem test_segmentation/test_sem_image/image2_sem
 
This will generate **'77_axon_morphometrics.xlsx'** and **'78_axon_morphometrics.xlsx'** morphometrics files in the **'image1_sem'** directory. And, the **'image_axon_morphometrics.xlsx'** and **'image2_axon_morphometrics.xlsx'** morphometrics files are generated in the **'image2_sem'** directory:: 

    --image1_sem/
    ---- 77.png
    ---- 77_seg-axon.png
    ---- 77_seg-axonmyelin.png
    ---- 77_seg-myelin.png
    ---- 77_axon_morphometrics.xlsx

    ---- 78.png
    ---- 78_seg-axon.png
    ---- 78_seg-axonmyelin.png
    ---- 78_seg-myelin.png
    ---- 78_axon_morphometrics.xlsx

    ---- pixel_size_in_micrometer.txt

    ...

    --image2_sem/
    ---- image.png
    ---- image_seg-axon.png
    ---- image_seg-axonmyelin.png
    ---- image_seg-myelin.png
    ---- image_axon_morphometrics.xlsx

    ---- image2.png
    ---- image2_seg-axon.png
    ---- image2_seg-axonmyelin.png
    ---- image2_seg-myelin.png
    ---- image2_axon_morphometrics.xlsx
    
    ---- pixel_size_in_micrometer.txt 
    
Axon Shape: Circle vs Ellipse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Circle 
^^^^^^
**Usage** ::

    axondeepseg_morphometrics -i test_segmentation/test_sem_image/image1_sem/77.png -a circle

**Studies using Circle as axon shape:**

* Duval et al: https://pubmed.ncbi.nlm.nih.gov/30326296/
* Salini et al: https://www.frontiersin.org/articles/10.3389/fnana.2017.00129/full

Ellipse
^^^^^^^
**Usage** ::

    axondeepseg_morphometrics -i test_segmentation/test_sem_image/image1_sem/77.png -a ellipse

**Studies using Ellipse as axon shape:**

* Payne et al: https://pubmed.ncbi.nlm.nih.gov/21381867/
* Payne et al: https://pubmed.ncbi.nlm.nih.gov/22879411/
* Fehily et al: https://pubmed.ncbi.nlm.nih.gov/30702755/


.. NOTE :: In the literature, both equivalent diameter and ellipse minor axis are used to compute the morphometrics. 
           Thus, depending on the usecase, the user is advised to choose axon shape accordingly.
           


Morphometrics file
~~~~~~~~~~~~~~~~~~

The resulting **'axon_morphometrics.csv/xlsx'** file will contain the following columns headings. Most of the metrics are computed using `skimage.measure.regionprops <https://scikit-image.org/docs/stable/api/skimage.measure.html#regionprops>`_. 

By default for axon shape, that is, `circle`, the equivalent diameter is used. For `ellipse` axon shape, minor axis is used as the diameter. The equivalent diameter is defined as the diameter of a circle with the same area as the region. 

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Field
     - Description
   * - x0
     - Axon X centroid position in pixels.
   * - y0
     - Axon Y centroid position in pixels.
   * - gratio
     - Ratio between the axon diameter and the axon+myelin (fiber) diameter (`gratio = axon_diameter / axonmyelin_diameter`).
   * - axon_area
     - Area of the axon region in :math:`{\mu}`\ m\ :sup:`2`\ .
   * - axon_perimeter
     - Perimeter of the axon object in :math:`{\mu}`\ m.
   * - myelin_area
     - Difference between axon+myelin (fiber) area and axon area in :math:`{\mu}`\ m\ :sup:`2`\ .
   * - axon_diameter
     - Diameter of the axon in :math:`{\mu}`\ m. 
   * - myelin_thickness
     - Half of the difference between the axon+myelin (fiber) diameter and the axon diameter in :math:`{\mu}`\ m.
   * - axonmyelin_area
     - Area of the axon+myelin (fiber) region in :math:`{\mu}`\ m\ :sup:`2`\ .
   * - axonmyelin_perimeter
     - Perimeter of the axon+myelin (fiber) object in :math:`{\mu}`\ m.
   * - solidity
     - Ratio of pixels in the axon region to pixels of the convex hull image.
   * - eccentricity
     - Eccentricity of the ellipse that has the same second-moments as the axon region.
   * - orientation
     - Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the axon region.

Jupyter notebooks
-----------------

Here is a list of useful Jupyter notebooks available with AxonDeepSeg:

* `00-getting_started.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/00-getting_started.ipynb>`_:
    Notebook that shows how to perform axon and myelin segmentation of a given sample using a Jupyter notebook (i.e. not using the command line tool of AxonDeepSeg). You can also launch this specific notebook without installing and/or cloning the repository by using the `Binder link <https://mybinder.org/v2/gh/neuropoly/axondeepseg/master?filepath=notebooks%2F00-getting_started.ipynb>`_.

* `01-performance_metrics.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/03-performance_metrics.ipynb>`_:
    Notebook that computes a large set of segmentation metrics to assess the axon and myelin segmentation quality of a given sample (compared against a ground truth mask). Metrics include sensitivity, specificity, precision, accuracy, Dice, Jaccard, F1 score, Hausdorff distance.

* `02-morphometrics_extraction.ipynb <https://github.com/neuropoly/axondeepseg/blob/master/notebooks/04-morphometrics_extraction.ipynb>`_:
    Notebook that shows how to extract morphometrics from a sample segmented with AxonDeepSeg. The user can extract and save morphometrics for each axon (diameter, solidity, ellipticity, centroid, ...), estimate aggregate morphometrics of the sample from the axon/myelin segmentation (g-ratio, AVF, MVF, myelin thickness, axon density, ...), and generate overlays of axon/myelin segmentation masks, colocoded for axon diameter.

.. NOTE ::
     To open a notebook, go to the notebooks/ subfolder of AxonDeepSeg and launch a particular notebook as follows::
     
         cd notebooks
         jupyter notebook name_of_the_notebook.ipynb 

.. WARNING ::
   The current models available for segmentation are trained for patches of 256x256 pixels for SEM and 512x512 pixels for TEM and BF. This means that your input image(s) should be at least 256x256 or 512x512 pixels in size **after the resampling to the target pixel size of the model you are using to segment**. 

   For instance, the TEM model currently available has a target resolution of 0.01 micrometers per pixel, which means that the minimum size of the input image (in micrometers) is 5.12x5.12.

   **Option:** If your image to segment is too small, you can use padding to artificially increase its size (i.e. add empty pixels around the borders).

Manual correction of segmentation masks
=======================================

If the segmentation with AxonDeepSeg does not give optimal results, you can try one of the following options:

**Option 1: manual correction of the segmentation mask with FSLeyes**

* In FSLeyes, you can make corrections on the myelin segmentation mask using the Edit mode in **Tools > Edit mode**.
* Then, use the **Fill Axons** function to automatically fill the axons and create a corrected axon+myelin mask.
* For a detailed procedure, please consult the following link: `Manual correction with FSLeyes <https://docs.google.com/document/d/1S8i96cJyWZogsMw4RrlQYwglcOWd3HrM5bpTOJE4RBQ/edit>`_.
* As a reference, you can find more informtations about the FSLeyes Edit mode in the `user guide <https://open.win.ox.ac.uk/pages/fsl/fsleyes/fsleyes/userdoc/editing_images.html>`_.

**Option 2: manual labelling with GIMP software**

* To create a new axon+myelin manual mask or to make manual correction on an existing segmentation mask, you can use the GIMP software (`Link for download <https://www.gimp.org/>`_).
* If you are making correction on an existing segmentation mask, note that when you launch a segmentation, in the folder output, you will also find the axon and myelin masks (with the suffixes **'_seg-axon.png'** and **'_seg-myelin.png'**). You can then manually correct the myelin mask and create a corrected axon+myelin mask.
* For a detailed procedure, please consult the following link: `Manual labelling with GIMP <https://docs.google.com/document/d/10E6gzMP6BNGJ_7Y5PkDFmum34U-IcbMi8AvRruhIzvM/edit>`_.

Training Models
===============

To train your own model for use in AxonDeepSeg, please refer to the `IVADOMED documentation <https://ivadomed.org/tutorials/two_class_microscopy_seg_2d_unet.html>`_ on model training for two-class microscopy images.

Help
====

Whether you are a newcomer or an experienced user, we will do our best to help and reply to you as soon as possible. Of course, please be considerate and respectful of all people participating in our community interactions.

* If you encounter difficulties during installation and/or while using AxonDeepSeg, or have general questions about the project, you can start a new discussion on `AxonDeepSeg GitHub Discussions forum <https://github.com/neuropoly/axondeepseg/discussions>`_. We also encourage you, once you've familiarized yourself with the software, to continue participating in the forum by helping answer future questions from fellow users!
* If you encounter bugs during installation and/or use of AxonDeepSeg, you can open a new issue ticket on the `AxonDeepSeg GitHub issues tracker <https://github.com/neuropoly/axondeepseg/issues>`_.

Citation
========

If you use this work in your research, please cite:

Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional neural networks. Scientific Reports, 8(1), 3816. `Link to the paper <https://doi.org/10.1038/s41598-018-22181-4>`_.
