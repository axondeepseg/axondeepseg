Introduction
===============================================================================
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin
sheaths from the spinal cord tissue.

AxonDeepSeg was developed by NeuroPoly, the neuroimagery laboratory of Polytechnique Montréal.

Getting Started
===============================================================================
The following lines will help you install all you need to ensure that AxonDeepSeg is working. Test data and
instructions are provided to help you use AxonDeepSeg.

Installing python
-------------------------------------------------------------------------------

First, you should make sure that Python 2.7 is installed on your computer. Run the following command in the terminal:

``python -V``

The version of python should be displayed in the terminal. If not, you have to install Python 2.7 on your computer.
To do that, you can follow the instructions given on
`the official python wiki <https://wiki.python.org/moin/BeginnersGuide/Download>`_.

Installing pip
-------------------------------------------------------------------------------

The second step is to ensure that pip is installed. Pip is a tool that helps you manage your python modules.
To check that, run the following command in your terminal:

``pip -V``

If the result of this command is the version of pip as well as its installation path, you're all set!
Else, you can follow the instructions to install it `at this URL <https://pip.pypa.io/en/stable/installing/>`_.

We also recommend to upgrade to the last version of pip:

``pip install --upgrade pip``

Installing git
-------------------------------------------------------------------------------

In order to download the software AxonDeepSeg, you also need to make sure that a recent version of git
is installed on your computer. Run:

``git``

in your terminal to check if git is installed. If there is an error or if you want to upgrade git to the last existing
version, you can follow the documentation on
`the official website <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.


Installing a virtual environment
-------------------------------------------------------------------------------

Before installing AxonDeepSeg, we will need to set up a virtual environment.
A virtual environment is a tool that lets you install specific versions of the python modules you want.
It will allow AxonDeepSeg to run with respect to its module requirements,
without affecting the rest of your python installation.

First, install the python module virtualenv via pip:

``pip install --upgrade virtualenv``

Once the module virtualenv is installed, you have to choose a folder where you virtual environments will be stored.
We suggest to install them in your home directory. We will continue this guide assuming that we store the
virtual environments in a directory in your home folder, but it is up to you to choose where you want to store them.

First, navigate to your home directory:

``cd ~``

We will now create a virtual environment. For clarity, we will name it ads_venv.

``virtualenv ads_venv``

To activate it, run the following command:

``source ads_venv/bin/activate``

If you performed all the steps correctly, your username in the console should now be preceded by the name of your
virtual environment between parenthesis, like this:

``(ads_venv) username@hostname /home/...``

To see which python modules are installed, you can run:

``pip list``


Installing AxonDeepSeg
-------------------------------------------------------------------------------

We are now going to install the software AxonDeepSeg.

.. CAUTION ::
   Make sure that the virtual environment is activated before you run the following command.

First, install all the modules required using the following command:

``pip install --upgrade numpy pandas tabulate pillow tensorflow tqdm mpld3 sklearn scikit-image scipy``

.. NOTE ::
   For information, here is a list of requirements for AxonDeepSeg. The program may work with earlier versions of
   these modules.
   * numpy==1.13.1
   * pandas==0.20.1
   * tabulate tabulate==0.7.7
   * Pillow==4.1.1
   * tensorflow==1.3.0
   * tqdm==4.14.0
   * mpld3==0.3
   * scikit-learn==0.18.1
   * scikit-image==0.13.0
   * scipy==0.19.0



Now is the time to clone the AxonDeepSeg project from GitHub. First, go to the place you want to download the template
to. In our case, we are going to use our home directory:


``cd ~``

You can now download the software by cloning the repository using the following command:

``git clone https://github.com/neuropoly/axondeepseg.git``

Once the download is done, go inside the downloaded directory.

``cd axondeepseg``

And install AxonDeepSeg by running the following command:

``pip install -e .``

AxonDeepSeg is now installed.

In order to test the installation and follow our guidelines, switch to the current branch:

``git checkout public_version``

Models
-------------------------------------------------------------------------------

Two models are available by default. You can download them `here <https://www.dropbox.com/sh/k71wnag0ztz0cpu/AADUGOC8SpLd7FWLtIBmVG7pa?dl=0>`_.

* A SEM model, that works at a resolution of 0.1 micrometer per pixel.
* A TEM model, that works at a resolution of 0.01 micrometer per pixel.

The easiest and most convenient solution is to put these two models in the models/ folder from AxonDeepSeg root directory.
Your architecture would thus look like this: ::

    axondeepseg/
    -- models/
    ---- defaults/
    ------ default_SEM_model_v1
    ------ default_TEM_model_v1

Acquisitions
-------------------------------------------------------------------------------

The acquisitions you want to segment must be stored following a particular architecture. ::

    acquisition_folder/
    -- acquisition.png
    -- pixel_size_in_micrometer.txt

.. NOTE ::
   The acquisitions must be saved in png format.

* The file *acquisition.png* is the image to segment.
* The file *pixel_size_in_micrometer.txt* contains a single float number corresponding to the resolution of the acquisition, that is the size of a pixel, in micrometer.


If you want to test AxonDeepSeg, you can download the test data available `here <https://www.dropbox.com/sh/xftifr8dr4je0o7/AADgF5l-2M4Z9WOdh9xvcVDva?dl=0>`_.


Using AxonDeepSeg
-------------------------------------------------------------------------------

To learn to use AxonDeepSeg, you will need some acquisitions image to segment. If you don't have some,
you can download the test data using the instructions in the Acquisitions part of this tutorial.

We are going to put the data in the data folder from AxonDeepSeg root directory. Note that this is not an obligation, as
you will be able to segment data even if it is not located inside the AxonDeepSeg directory.

Once you have downloaded the default models and the test data, go to the AxonDeepSeg folder in
the axondeepseg root directory. In our case:

``cd ~/projects/axondeepseg/AxonDeepSeg``

The script to launch is called *segment.py*. It takes several arguments:

* t: type of the acquisition. SEM or TEM.
* p: path to the acquisition.
* v: (optional) verbosity level. Default 0.

    * 0 displays only a progress bar indicating the advancement of the segmentations.
    * 2 displays information about the current step of the segmentation of the current acquisition.

* o: (optional) overlap value. Number of pixel to use when overlapping predictions. The higher, the more time the segmentation will take. Default 25.

To segment the tem acquisition we just downloaded with a detail of the steps of the segmentation, run the following command:

``python segment.py -t SEM -p ../data/test_segmentation/test_sem_image/image1_sem/77.png -v 2``

The script will automatically read the acquisition resolution.
The different steps will be displayed in the terminal thanks to the verbosity level set to 2.
The segmented acquisition itself will be saved in the same folder as the acquisition image, with the prefix 'segmentation_', in png format.

You can also decide to segment multiple acquisitions at the same time.
In that case, each acquisition must be located in its own folder.
Each folder must hence contain at minimum a .png acquisition image to segment, and a pixel_size_in_micrometer.txt file
where the resolution of the acquisition is stored, in micrometer per pixel.
All these acquisitions folders must then be located in the same global folder.

When using the segmentation script, you then just have to indicate the path to the global folder, like this:

``python segment.py -t SEM -p ../data/test_segmentation/test_sem_image/``

This line will segment all acquisitions in acquisition folders contained in the directory test_sem_image.
Each segmentation will be saved in the same folder as its corresponding acquisition.

.. NOTE ::
   When looking in an acquisition folder for an acquisition to segment, the script will first look for an image named
   'image.png'. If found, it will segment it. Else, it will segment the first .png file which name does not begin with
   'segmentation_'.

Finally, you can segment multiple images and folders at the same time, using the following command:

``python segment.py -t SEM -p ../data/test_segmentation/test_sem_image/ ../data/test_segmentation/test_sem_image_2/image2_sem/95.png -o 40``

The previous command will segment all the acquisitions in the folders located in the test_sem_image directory,
as well as the acquisition 95.png, with an overlap value of 40 pixels.

Licensing
===============================================================================

MIT.

Acknowledgements
===============================================================================
todo