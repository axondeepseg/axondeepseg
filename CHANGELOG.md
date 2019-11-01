Changelog
===============================================================================

Version [2.2dev] - XXXX-XX-XX
-------------------------------------------------------------------------------

**Changed:**
- Refractored data augmentation to use `Albumentation` library.
- Cleaned jupyter notebooks and reduced from 13 notebooks to 4.
- Switched to `Dice Loss` from `Categorical Cross Entropy` as loss function.
- Updated SEM and TEM models for better performance.
- Shifted AxonDeepSeg from TensorFlow to Keras framework.
- Upgraded CUDA to 10.0 and tensorflow to 1.13.1.
- Resolve image rescale warnings
- Handle exception for images smaller than minimum patch size after resizing
- Revert tensorflow requirekment to 1.3.0 and remove tifffile requirement
- Remove `matplotlib.pyplot` from source code and refactor to full OO plotting
- Standardize path management to `pathlib` library
- Shifted AxonDeepSeg from TensorFlow to Keras framework.
- Upgraded CUDA to 10.0 and tensorflow to 1.13.1.
– Add FSLeyes plugin



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
