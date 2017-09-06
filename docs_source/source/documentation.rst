Introduction
===============================================================================
AxonDeepSeg is an open-source software using deep learning and aiming at automatically segmenting axons and myelin
sheaths from the spinal cord tissue.

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

Acquisitions
-------------------------------------------------------------------------------

The acquisitions you want to segment must be stored following a particular architecture::

    acquisitions_to_segment/
    --acquisition1_folder/
    ---- acquisition.png
    ---- pixel_size_in_micrometer.txt
    --acquisition2_folder/
    ---- acquisition.png
    ---- pixel_size_in_micrometer.txt
    ...

.. NOTE ::
   The acquisitions must be saved in **png format**.

* The file *acquisition.png* is the image to segment.
* The file *pixel_size_in_micrometer.txt* contains a single float number corresponding
to the resolution of the acquisition, that is the size of a pixel, in micrometer.

If you want to test AxonDeepSeg, you can download the test data available
`here <https://osf.io/rtbwc/download>`_.


Using AxonDeepSeg
-------------------------------------------------------------------------------

To learn to use AxonDeepSeg, you will need some acquisitions image to segment. If you don't have some,
you can download the test data using the instructions in the `Acquisitions` part of this tutorial.

Once you have downloaded the test data, go to the extracted test data folder. In our case::

    cd test_segmentation

The script to launch is called **axondeepseg**. It takes several arguments:

* **t**: type of the acquisition. SEM or TEM.
* **p**: path to the acquisition.
* **v**: (optional) verbosity level. Default 0.

    * 0 displays only a progress bar indicating the advancement of the segmentations.
    * 1 displays additionally the path of the image that was just segmented.
    * 2 displays additionally information about the current step of the segmentation of the current acquisition.

To segment one of the image that we downloaded (here, an SEM image), run the following command::

    axondeepseg -t SEM -p test_sem_image/image1_sem/77.png -v 2

The script will automatically read the acquisition resolution encoded in the file: *pixel_size_in_micrometer.txt*
The different steps will be displayed in the terminal thanks to the verbosity level set to 2.
The segmented acquisition itself will be saved in the same folder as the acquisition image, with the prefix 'segmentation_', in png format.

To segment multiple images, please use the folder structure explained above, i.e., each image should be in a separate folder. Then, the path to segment will be the parent folder that contains each separate sub-folder::

    axondeepseg -t SEM -p test_sem_image/

Each output segmentation will be saved in the corresponding sub-folder.

.. NOTE ::
   If there is multiple .png files inside a sub-folder, then AxonDeepSeg will ignore all files which names contains 'segmentation', so that if you want to re-run a segmentation, it will select the image.

Finally, you can segment multiple images and folders at the same time, using the following command::

    axondeepseg -t SEM -p test_sem_image/ test_sem_image_2/image1_sem/95.png -o 40

The previous command will segment all the acquisitions in the folders located in the test_sem_image directory,
as well as the acquisition 95.png, with an overlap value of 40 pixels.

Licensing
===============================================================================

MIT.

Acknowledgements
===============================================================================
