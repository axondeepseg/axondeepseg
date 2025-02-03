Changelog
===============================================================================

## v5.0.0 (2025-01-23)
[View detailed changelog](https://github.com/axondeepseg/axondeepseg/compare/v4.1.0...v5.0.0)

**BUG**

 - Fix 16 bit conversion bug.  [View pull request]( https://github.com/axondeepseg/axondeepseg/pull/710)
 - Fix windows install bug.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/734)
 - Fix a bug where the error message would not show up if the model couldn't be applied.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/750)
 - Change imageio as_gray to mode=L due to imageio v2->v3.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/739)
 - Remove unicode in units names.  [View pull request](ttps://github.com/axondeepseg/axondeepseg/pull/780)
 - Fix download_models bug - download to path.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/836)
 - Update _widget.py [View pull request](https://github.com/axondeepseg/axondeepseg/pull/861)

**ENHANCEMENT**

 - Add warnings related to "no-patch" option.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/704)
 - Change pixel size to reduce RAM usage during no-patch test.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/707)
 - Support unmyelinated axons in morphometrics.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/796)
 - CLI: Log/print commit hash, and --version option.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/847)

**FEATURE**

 - Segment images without patches.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/696)
 - Add GPU ID setting in CLI and FSLeyes.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/701)
- `ivadomed` to `nnunetv2` backend migration.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/800)
 - Log CLI args, add breakline [View pull request](https://github.com/neuropoly/axondeepseg/pull/854)


**DOCUMENTATION**

 - Fix broken link.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/720)
 - Patch RTD setup to deprication.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/762)
 - Add testimonial section in documentation.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/764)
 - Add A. Wheeler testimonial.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/767)
 - Add morphometrics algorithms description.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/777)
 - Update sphinx to fix doc build.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/816)
 - Fix youtube bug in docs.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/825)
 - Set `fail_on_warning = True` and resolve documentation warnings to ensure docs pass.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/846)
 - Add Napari Plugin animation in doc.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/850)
 - README: Add dark/light mode compatibility.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/853)
 - Add reinstallation documentation [View pull request](https://github.com/neuropoly/axondeepseg/pull/788)
 - Add preprint link to model_cards [View pull request](https://github.com/neuropoly/axondeepseg/pull/851)
 - Fix crosslink bug in documentation [View pull request](https://github.com/neuropoly/axondeepseg/pull/858)
 - Add generalist preprint in readme [View pull request](https://github.com/neuropoly/axondeepseg/pull/859)

**GUI**

 - Add no patch option to the FSLeyes plugin settings menu.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/700)
 - Add napari plugin, set as default GUI tool.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/726)
 - fix napari plugin not launching properly on Windows.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/828)

**INSTALLATION**

 - Update IVADOMED version.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/693)
 - Update ivadomed version to 2.9.8.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/708)
 - remove openCV requirement.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/713)
 - [patch] remove matplotlib version requirement.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/729)
 - Fix imagio below latest version by @mathieuboudreau in https://github.com/axondeepseg/axondeepseg/pull/740
 - Remove conda-forge channel for speed and refactor environment.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/756)
 - Patch onnxissue 1.16.0 issue.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/758)
 - Update ivadomed version to 2.9.10.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/792)
 - Pin acvl_utils!=0.2.1.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/834)
 - Pin skimage<0.25.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/848)
 - Remove scikitlearn pinned version and use updated reference pickle for morphometrics data testing   [View pull request](https://github.com/axondeepseg/axondeepseg/pull/860)

**REFACTORING**

 - Resolves warnings.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/694)
 - Remove deprecated file and folder.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/721)
 - Change "download_models" to "download_model".  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/830)
 - Class value check - Change error to warning.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/838)
 - Refactor config.py into ADS/params.py.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/844)

**TESTING**

 - Improve and force imread/imwrite conversion to 8bit int.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/669)
 - Github Actions failure: macos " No installed conda 'base' environment found at !".  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/801)
 - Consolidate coveralls services.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/819)
 - Add CLI option to run all tests.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/845)
 - Update imread/imwrite functionality and tests  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/855)
 - Change test image(s) in rtd [View pull request](https://github.com/axondeepseg/axondeepseg/pull/852)

## v4.1.0 (2022-10-07)
[View detailed changelog](https://github.com/axondeepseg/axondeepseg/compare/v4.0.0...v4.1.0)

**BUG**

 - Resolve "tif in filename" imread bug & refactor valid image exensions.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/629)
 - Fix morphometrics bug when writing index/axonmyelin_index files.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/622)

**ENHANCEMENT**

 - `--border-info` option for morphometrics (optical fractionator technique).  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/650)
 - Add border_touching flag to morphometrics.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/645)
 - Zoom factor sweep.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/632)
 - Zoom factor option for segmentation via CLI.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/620)

**FEATURE**

 - Save instance segmentation image.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/667)
 - Add logging for segmentation/morphometrics.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/633)
 - Zoom factor sweep.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/632)

**DOCUMENTATION**

 - Add model repo links in the doc.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/621)

**INSTALLATION**

 - Simplify specifying torch dependency.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/642)
 - Update ivadomed to 2.9.5.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/625)
 - Update ivadomed to 2.9.4.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/612)

**TESTING**

 - Add test for 16bit TIF grayscale file.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/630)
 - `environment.yaml`: Pin `pillow!=9.0.0`.  [View pull request](https://github.com/axondeepseg/axondeepseg/pull/602)

## Version 4.0.0 - 2022-02-03
[View detailed changelog](https://github.com/neuropoly/axondeepseg/compare/v3.3.0...v4.0.0)

**NOTICE**
 - Due to the change in implementations on an upgraded dependency (skimage v0.14.2 -> v0.18.3), the morphometrics values "solidity" and "orientation" differ from values produced by AxonDeepSeg v3.x. "solidity" values produced can vary slightly due to numerical precision in algorithmic changes in skimage (on the order of 1%), a the convention for "orientation" changed from "rc" to "xy", leading to a difference in value by pi/2. See this issue for more details: https://github.com/neuropoly/axondeepseg/issues/589

**BUG** 
 - Fix NaNs not appearing when generating morphometrics with the GUI [View pull request](https://github.com/neuropoly/axondeepseg/pull/592)
 
**FEATURE**

 - Integrate IVADOMED into project [View pull request](https://github.com/neuropoly/axondeepseg/pull/547)
 - Changed all instances of imageio_imread by ads_utils.imread [View pull request](https://github.com/neuropoly/axondeepseg/pull/5927)

**INSTALLATION**

 - Added Mac M1 compatibility  [View pull request](https://github.com/neuropoly/axondeepseg/pull/547)

## Version 3.3.0 - 2022-01-14
[View detailed changelog](https://github.com/neuropoly/axondeepseg/compare/v3.2.0...v3.3.0)

**BUG**

 - Fix morphometrics error with ellipse axon shape [View pull request](https://github.com/neuropoly/axondeepseg/pull/558)
 - Fix patches2im_overlap when width = height  [View pull request](https://github.com/neuropoly/axondeepseg/pull/510)
 - Patch myelin thickness calculation [View pull request](https://github.com/neuropoly/axondeepseg/pull/449)
 - Bug fix due to h5py package version in fsleyes [View pull request](https://github.com/neuropoly/axondeepseg/pull/392)
 - Pin h5py~=2.10.0 as a workaround for upstream Keras/TensorFlow issues [View pull request](https://github.com/neuropoly/axondeepseg/pull/382)
 - Resolve bug that makes TEM data not segment well with multiple files [View pull request](https://github.com/neuropoly/axondeepseg/pull/370)

**ENHANCEMENT**

 - Added the axon shape selection in the GUI (take two) [View pull request](https://github.com/neuropoly/axondeepseg/pull/541)
 - Generate numbers image with CLI  [View pull request](https://github.com/neuropoly/axondeepseg/pull/519)
 - Refactor the code for the settings menu [View pull request](https://github.com/neuropoly/axondeepseg/pull/507)
 - Add Optical Microscopy model in notebooks [View pull request](https://github.com/neuropoly/axondeepseg/pull/530)
 - Add perimeter to morphometrics[View pull request](https://github.com/neuropoly/axondeepseg/pull/501)
 - Add CLI support to generate morphometrics file. [View pull request](https://github.com/neuropoly/axondeepseg/pull/434)
 - Replaced rcParams with font_manager to find the font [View pull request](https://github.com/neuropoly/axondeepseg/pull/491)
 - Remove wildcard imports [View pull request](https://github.com/neuropoly/axondeepseg/pull/487)
 - Migration to GitHub Actions [View pull request](https://github.com/neuropoly/axondeepseg/pull/479)
 - Change the name of -o flag to -overlap flag  [View pull request](https://github.com/neuropoly/axondeepseg/pull/474)
 - Add a model to .gitignore [View pull request](https://github.com/neuropoly/axondeepseg/pull/471)
 - Change the default name for saving the morphometrics file [View pull request](https://github.com/neuropoly/axondeepseg/pull/472)
 - Use pathlib in the ads_plugin instead of os.path [View pull request](https://github.com/neuropoly/axondeepseg/pull/448)
 - Integrate Simeon's model with FSLeyes and CLI [View pull request](https://github.com/neuropoly/axondeepseg/pull/457)
 - Fix Naming Convention [View pull request](https://github.com/neuropoly/axondeepseg/pull/441)
 -  Fixed Notebooks and Binder links in doc[View pull request](https://github.com/neuropoly/axondeepseg/pull/413)
 - Move test files to OSF, add download functionality [View pull request](https://github.com/neuropoly/axondeepseg/pull/373)
 - Add postprocessing tests [View pull request](https://github.com/neuropoly/axondeepseg/pull/365)

**FEATURE**

 - Implement Batch morphometrics for images present in directory/directories [View pull request](https://github.com/neuropoly/axondeepseg/pull/518)
 - Implement Ellipse Minor Axis as Diameter [View pull request](https://github.com/neuropoly/axondeepseg/pull/399)
 - Add a settings menu to the FSLeyes plugin  [View pull request](https://github.com/neuropoly/axondeepseg/pull/462)
 - Add Axon mask simulator [View pull request](https://github.com/neuropoly/axondeepseg/pull/179)

**DOCUMENTATION**

 - Update Youtube video links [View pull request](https://github.com/neuropoly/axondeepseg/pull/574)
 - Apply RTD fix to make builds pass again  [View pull request](https://github.com/neuropoly/axondeepseg/pull/578)
 - Update figure link  [View pull request](https://github.com/neuropoly/axondeepseg/pull/575)
 - Migrate images to new repo [View pull request](https://github.com/neuropoly/axondeepseg/pull/543)
 - Change link [View pull request](https://github.com/neuropoly/axondeepseg/pull/535)
 - Add description of morphometrics columns headings [View pull request](https://github.com/neuropoly/axondeepseg/pull/516)
 - Add a reference to Readme.md [View pull request](https://github.com/neuropoly/axondeepseg/pull/500)
 - Update the documentation for installing the GPU compatible version of ADS [View pull request](https://github.com/neuropoly/axondeepseg/pull/490)
 - Add tutorial video link in RTD  [View pull request](https://github.com/neuropoly/axondeepseg/pull/493)
 - Fix docs for optical microscopy model [View pull request](https://github.com/neuropoly/axondeepseg/pull/485)
 - Documentation updates  [View pull request](https://github.com/neuropoly/axondeepseg/pull/433)
 - Fix new changelog display in RTD  [View pull request](https://github.com/neuropoly/axondeepseg/pull/388)
 - Add zoomed window on masks examples in RTD  [View pull request](https://github.com/neuropoly/axondeepseg/pull/386)
 - Remove PyPI install instructions from RTD [View pull request](https://github.com/neuropoly/axondeepseg/pull/376)
 - Fix youtube video link in RTD [View pull request](https://github.com/neuropoly/axondeepseg/pull/379)

**INSTALLATION**

 - Link the test_files to GitHub [View pull request](https://github.com/neuropoly/axondeepseg/pull/534)
 - Download models from Github repo [View pull request](https://github.com/neuropoly/axondeepseg/pull/533)
 - Generate images synthetically in tests [View pull request](https://github.com/neuropoly/axondeepseg/pull/520)
 - Change install to one env file & update docs  [View pull request](https://github.com/neuropoly/axondeepseg/pull/484)
 - Refactor AxonDeepSeg/models directory and update .gitignore [View pull request](https://github.com/neuropoly/axondeepseg/pull/476)
 - Change how ADS dependencies are installed [View pull request](https://github.com/neuropoly/axondeepseg/pull/452)
 - Add PR template [View pull request](https://github.com/neuropoly/axondeepseg/pull/467)
 - Remove old config file (set_config) [View pull request](https://github.com/neuropoly/axondeepseg/pull/456)


## Version 3.2.0 - 2020-10-16
[View detailed changelog](https://github.com/neuropoly/axondeepseg/compare/v3.0...v3.2.0)

**BUG**

 - Fix redownloading models bug.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/322)
 - [Bug] Resolve segment folder bug when using relative paths.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/320)
 - Resolve bug that makes TEM data not segment well (#293 and #249).  [View pull request](https://github.com/neuropoly/axondeepseg/pull/294)
 - Add missing import to apply_model.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/290)

**ENHANCEMENT**

 - FSLeyes plugin: default morphometrics file extension and font size .  [View pull request](https://github.com/neuropoly/axondeepseg/pull/358)
 - Remove gaussian blur option for data augmentation.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/332)
 - Remove overlap when saving masks.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/328)
 - Resume training from checkpoint.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/301)

**FEATURE**

 - Add an "Axon numbers" overlay in FSLeyes.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/342)
 - [FSLeyes plugin] changed Image.save/open with ads_utils.read/write.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/296)

**DOCUMENTATION**

 - Add warning to doc for re-download of model folders when re-installing.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/351)
 - Update documentation for manual masks creation.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/338)
 - Add resampled_resolutions parameter in getting_started notebook.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/325)
 - [Doc] Add notice about restarting FSLeyes to see the plugin for installation.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/317)
 - Fix format in Changelog.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/291)

**INSTALLATION**

 - Fix for the Windows installation.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/362)
 - Move test images in default models folder.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/344)
 - Upgrade to Python 3.7.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/307)
 - Use a specific version of FSLeyes for the plugin.  [View pull request](https://github.com/neuropoly/axondeepseg/pull/305)

Version [3.0] - 2020-03-13
-------------------------------------------------------------------------------

**Changed:**

- Refractored data augmentation to use `Albumentation` library.
- Cleaned jupyter notebooks and reduced from 13 notebooks to 5.
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
- Add FSLeyes plugin

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
- Updated documentation on the usage of ads_base.
- Change of axon and myelin masks filenames for better clarity.

Version [0.3] - 2018-02-22
-------------------------------------------------------------------------------

**Added:**

- Compatibility for image inputs other than png
- Pre-processing of input images is now done inside AxonDeepSeg

**Changed:**

- Help display when running AxonDeepSeg from terminal
