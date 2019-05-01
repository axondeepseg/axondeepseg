"""
This is an FSLeyes plugin script that integrates AxonDeepSeg tools into FSLeyes.

Author : Stoyan I. Asenov
"""

import wx

import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.actions.loadoverlay as ovLoad

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageOps
import scipy.misc
import os
import json
from pathlib import Path

import AxonDeepSeg
from AxonDeepSeg.apply_model import axon_segmentation
import AxonDeepSeg.morphometrics.compute_morphometrics as compute_morphs

import math
from scipy import ndimage as ndi
from skimage import measure, morphology, feature


class ADScontrol(ctrlpanel.ControlPanel):
    """
    This class is the object corresponding to the AxonDeepSeg control panel.
    """

    def __init__(self, ortho, *args, **kwargs):
        """
        This function initializes the control panel. It generates the widgets and adds them to the panel. It also sets
        the initial position of the panel to the left
        :param ortho: This is used to access the ortho ops in order to turn off the X and Y canvas as well as the cursor
        :param args:
        :param kwargs:
        """
        ctrlpanel.ControlPanel.__init__(self, ortho, *args, **kwargs)

        # Add a sizer to the control panel
        # This sizer will contain the buttons
        sizerH = wx.BoxSizer(wx.VERTICAL)


        # Add the logo to the control panel
        ADS_logo = self.getLogo()
        sizerH.Add(ADS_logo, flag=wx.SHAPED, proportion=1)

        # Add the citation to the control panel
        citationBox = wx.TextCtrl(self, value=self.getCitation(), size=(100, 50), style=wx.TE_MULTILINE)
        sizerH.Add(citationBox, flag=wx.SHAPED, proportion=1)

        # Add the image loading button
        loadPng_button = wx.Button(self, label="Load PNG or TIF file")
        loadPng_button.Bind(wx.EVT_BUTTON, self.onLoadPngButton)
        loadPng_button.SetToolTip(wx.ToolTip("Loads a .png or .tif file into FSLeyes"))
        sizerH.Add(loadPng_button, flag=wx.SHAPED, proportion=1)

        # Add the save Segmentation button
        saveSegmentation_button = wx.Button(self, label="Save segmentation")
        saveSegmentation_button.Bind(wx.EVT_BUTTON, self.onSaveSegmentation_button)
        saveSegmentation_button.SetToolTip(wx.ToolTip("Saves the axon and myelin masks in the selected folder"))
        sizerH.Add(saveSegmentation_button, flag=wx.SHAPED, proportion=1)

        # Add the model choice combobox
        self.modelChoices = ['SEM', 'TEM', 'other']
        self.modelCombobox = wx.ComboBox(self, choices=self.modelChoices, size=(100, 20), value='Select the modality')
        self.modelCombobox.SetToolTip(wx.ToolTip("Select the modality used to acquire the image"))
        sizerH.Add(self.modelCombobox, flag=wx.SHAPED, proportion=1)

        # Add the button that applies the prediction model
        applyModel_button = wx.Button(self, label='Apply ADS prediction model')
        applyModel_button.Bind(wx.EVT_BUTTON, self.onApplyModel_button)
        applyModel_button.SetToolTip(wx.ToolTip("Applies the prediction model and displays the masks"))
        sizerH.Add(applyModel_button, flag=wx.SHAPED, proportion=1)

        # Add the button that runs the watershed algorithm
        runWatershed_button = wx.Button(self, label='Run Watershed')
        runWatershed_button.Bind(wx.EVT_BUTTON, self.onRunWatershed_button)
        runWatershed_button.SetToolTip(wx.ToolTip("Uses a watershed algorithm to find the different axon+myelin"
                                                  "objects. This is used to see if where are connections"
                                                  " between two axon+myelin objects."))
        sizerH.Add(runWatershed_button, flag=wx.SHAPED, proportion=1)

        # Add the fill axon tool
        fillAxons_button = wx.Button(self, label='Fill axons')
        fillAxons_button.Bind(wx.EVT_BUTTON, self.onFillAxons_button)
        fillAxons_button.SetToolTip(wx.ToolTip("Automatically fills the axons inside myelin objects."
                                               " THE MYELIN OBJECTS NEED TO BE CLOSED AND SEPARATED FROM EACH "
                                               "OTHER (THEY MUST NOT TOUCH) FOR THIS TOOL TO WORK CORRECTLY."))
        sizerH.Add(fillAxons_button, flag=wx.SHAPED, proportion=1)

        # Set the sizer of the control panel
        self.SetSizer(sizerH)

        # Initialize the variables that are used to track the active image/masks
        self.imageDirPath = None
        self.mostRecentMyelinMaskName = None
        self.mostRecentAxonMaskName = None
        self.mostRecentWatershedMaskName = None

        # Toggle off the X and Y canvas
        oopts = ortho.sceneOpts
        oopts.showXCanvas = False
        oopts.showYCanvas = False

        # Toggle off the cursor
        oopts.showCursor = False

    def onLoadPngButton(self, event):
        """
        This function is called when the user presses on the Load Png button. It allows the user to select a PNG or TIF
        image, convert it into a NIfTI and load it into FSLeyes.
        """
        # Ask the user which file he wants to convert
        with wx.FileDialog(self, "select Image file",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:  # The user cancelled the operation
                return

            inFile = fileDialog.GetPath()

        # Check if the image format is valid
        if (inFile[-4:] != '.png') and (inFile[-4:] != '.tif'):
            print('Invalid file extension')
            return

        # Store the directory path and image name for later use in the application of the prediction model
        self.imageDirPath = os.path.dirname(inFile)
        self.pngImageName = inFile[self.imageDirPath.__len__() + 1:]

        # If a watershed mask is already loaded, make sure it isn't used with the newly loaded image
        self.mostRecentWatershedMaskName = None

        # Call the function that convert and loads the png or tif image
        self.loadPngImageFromPath(inFile)

    def onApplyModel_button(self, event):
        """
        This function is called when the user presses on the ApplyModel button. It is used to apply the prediction model
        selected in the combobox. The segmentation masks are then loaded into FSLeyes
        """

        # Check if an image has been loaded
        if self.imageDirPath is None:
            print('Please load a PNG file')
            return

        selectedModel = self.modelCombobox.GetStringSelection()


        # Get the path of the selected model
        if selectedModel == 'other':
            # Ask the user where the model is located
            with wx.DirDialog(self, "select the directory in which the model is located", defaultPath="",
                              style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as fileDialog:

                if fileDialog.ShowModal() == wx.ID_CANCEL:  # The user cancelled the operation
                    return

            modelPath = fileDialog.GetPath()


        elif (selectedModel == 'SEM') or (selectedModel == 'TEM'):
            modelPath = os.path.dirname(AxonDeepSeg.__file__)
            modelPath = os.path.join(modelPath, 'models', 'default_' + selectedModel + '_model_v1')

        else:
            print('Please select a model')
            return

        # Check if the pixel size txt file exist in the imageDirPath
        pixelSizeExists = os.path.isfile(self.imageDirPath + '/pixel_size_in_micrometer.txt')

        # if it doesn't exist, ask the user to input the pixel size and create the .txt file
        if pixelSizeExists is False:
            with wx.TextEntryDialog(self, "Enter the pixel size in micrometer", value="0.07") as textEntry:
                if textEntry.ShowModal() == wx.ID_CANCEL:
                    return

                pixelSizeStr = textEntry.GetValue()
            textFile = open(self.imageDirPath + '/pixel_size_in_micrometer.txt', 'w')
            textFile.write(pixelSizeStr)
            textFile.close()

        # Load model configs and apply prediction
        model_configfile = os.path.join(modelPath, 'config_network.json')
        with open(model_configfile, 'r') as fd:
            config_network = json.loads(fd.read())

        prediction = axon_segmentation([self.imageDirPath], [self.pngImageName], modelPath,
                                       config_network, verbosity_level=3)
        # The axon_segmentation function creates the segmentation masks and stores them as PNG files in the same folder
        # as the original image file.

        # Load the axon and myelin masks into FSLeyes
        axonMaskPath = self.imageDirPath + '/AxonDeepSeg_seg-axon.png'
        myelinMaskPath = self.imageDirPath + '/AxonDeepSeg_seg-myelin.png'
        self.loadPngImageFromPath(axonMaskPath, isMask=True, colormap='blue')
        self.loadPngImageFromPath(myelinMaskPath, isMask=True, colormap='red')

        # Update the active masks
        self.mostRecentAxonMaskName = 'AxonDeepSeg_seg-axon'
        self.mostRecentMyelinMaskName = 'AxonDeepSeg_seg-myelin'

    def onSaveSegmentation_button(self, event):
        """
        This function saves the active myelin and axon masks as PNG images. Three (3) images are generated in a folder
        selected by the user : one with the axon mask, one with the myelin mask and one with both.
        """

        # Check if a mask is loaded
        if (self.mostRecentMyelinMaskName is None) or (self.mostRecentAxonMaskName is None):
            'Masks not found'
            return

        # Ask the user where to save the segmentation
        with wx.DirDialog(self, "select the directory in which the segmentation will be save", defaultPath="",
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

        saveDir = fileDialog.GetPath()

        # Find most recent segmentation masks overlays within the overlay list and store them in a variable as numpy
        # arrays.
        # Note: since PIL uses a different convention for the X and Y coordinates, some array manipulation has to be
        # done.
        for anImage in self.overlayList:
            if anImage.name == self.mostRecentMyelinMaskName:
                myelinArray = np.array(anImage[:, :, 0] * 255, copy=True, dtype=np.uint8)
                myelinArray = np.flipud(np.rot90(myelinArray, k=3, axes=(0, 1)))
            if anImage.name == self.mostRecentAxonMaskName:
                axonArray = np.array(anImage[:, :, 0] * 255, copy=True, dtype=np.uint8)
                axonArray = np.flipud(np.rot90(axonArray, k=3, axes=(0, 1)))

        # Save the arrays as PNG files
        myelinAndAxonImage = Image.fromarray((myelinArray // 2 + axonArray).astype(np.uint8))
        myelinAndAxonImage.save(saveDir + '/ADS_seg.png')
        myelinImage = Image.fromarray(myelinArray)
        myelinImage.save(saveDir + '/ADS_seg-myelin.png')
        axonImage = Image.fromarray(axonArray)
        axonImage.save(saveDir + '/ADS_seg-axon.png')

    def onRunWatershed_button(self, event):
        """
        This function is called then the user presses on the runWatershed button. This creates a watershed mask that is
        used to locate where are the connections between the axon-myelin objects.
        """

        if self.mostRecentAxonMaskName is None:
            print('No axon mask loaded')
            return

        # Find the most recent axon and myelin masks

        axonMaskOverlay = None
        watershedMaskOverlay = None

        for anOverlay in self.overlayList:
            if anOverlay.name == self.mostRecentAxonMaskName:
                axonMaskOverlay = anOverlay

            if anOverlay.name == self.mostRecentMyelinMaskName:
                myelinMaskOverlay = anOverlay

            if (self.mostRecentWatershedMaskName is not None) and (anOverlay.name == self.mostRecentWatershedMaskName):
                watershedMaskOverlay = anOverlay

        if axonMaskOverlay is None:
            print("Couldn't find an axon mask.")
            return

        # Extract the data from the overlays
        axonArray = axonMaskOverlay[:, :, 0]
        myelinArray = myelinMaskOverlay[:, :, 0]

        # Compute the watershed mask
        watershedData = self.getWatershedSegmentation(axonArray, myelinArray)

        if self.mostRecentWatershedMaskName is None:
            # Save the watershed mask as a png then load it as an overlay
            watershedImageArray = np.flipud(np.rot90(watershedData, k=3, axes=(0, 1)))
            watershedImage = Image.fromarray(watershedImageArray)
            fileName = self.imageDirPath + '/watershed_mask.png'
            watershedImage.save(fileName)
            watershedMaskOverlay = self.loadPngImageFromPath(fileName, addToOverlayList=False)
            watershedMaskOverlay[:, :, 0] = watershedData
            self.overlayList.append(watershedMaskOverlay)

            # Apply a "random" colour mapping to the watershed mask
            opts = self.displayCtx.getOpts(watershedMaskOverlay)
            opts.cmap = 'random'

            self.mostRecentWatershedMaskName = 'watershed_mask'

        elif watershedMaskOverlay is not None:
            # Update the current watershed mask
            watershedMaskOverlay[:, :, :] = watershedData

    def onFillAxons_button(self, event):
        """
        This function is called when the fillAxon button is pressed by the user. It uses a flood fill algorithm to fill
        the inside of the myelin objects with the axon mask
        """
        # Find the most recent axon and myelin masks
        axonMaskOverlay = None
        myelinMaskOverlay = None

        for anOverlay in self.overlayList:
            if anOverlay.name == self.mostRecentAxonMaskName:
                axonMaskOverlay = anOverlay

            if anOverlay.name == self.mostRecentMyelinMaskName:
                myelinMaskOverlay = anOverlay

        if (axonMaskOverlay is None) or (myelinMaskOverlay is None):
            print("Couldn't find an axon or myelin mask.")
            return

        # Extract the data from the overlays
        axonArray = axonMaskOverlay[:, :, 0]
        myelinArray = myelinMaskOverlay[:, :, 0]

        # Get the centroid indexes
        centroidIndexMap = self.getMyelinCentroids(myelinArray)


        # Create an image with the myelinMask and floodfill at the coordinates of the centroids
        # Note: The floodfill algorithm only works on PNG images. Thus, the mask must be colorized before applying
        # the floodfill. Then, the array corresponding to the floodfilled color can be extracted.
        myelinImage = Image.fromarray(myelinArray * 255)
        myelinImage = ImageOps.colorize(myelinImage, (0, 0, 0, 255), (255, 255, 255, 255))
        for i in range(len(centroidIndexMap[0])):
            ImageDraw.floodfill(myelinImage, xy=(centroidIndexMap[1][i], centroidIndexMap[0][i]),
                                value=(127, 127, 127, 255))

        # Extract the axonArray and update the axon mask overlay
        axonExtractedArray = np.array(myelinImage.convert('LA'))
        axonExtractedArray = axonExtractedArray[:, :, 0]
        axonExtractedArray = np.equal(axonExtractedArray, 127 * np.ones_like(axonExtractedArray))
        axonExtractedArray = axonExtractedArray.astype(np.uint8)

        axonMaskOverlay[:, :, :] = axonExtractedArray

    def getMyelinCentroids(self, im_myelin):
        """
        This function is used to find the centroids of the myelin mask
        :param im_myelin:
        :type im_myelin: np.array
        :return: a list containing the coordinates of the centroid of every myelin object
        :rtype: list of int
        """
        # Label each myelin object
        im_myelin_label = measure.label(im_myelin)
        # Find the centroids of the myelin objects
        myelin_objects = measure.regionprops(im_myelin_label)
        ind_centroid = ([int(props.centroid[0]) for props in myelin_objects],
                        [int(props.centroid[1]) for props in myelin_objects])
        return ind_centroid

    def getWatershedSegmentation(self, im_axon, im_myelin, returnCentroids=False):
        """
        Parts of this function were copied from the code found in this document :
        https://github.com/neuropoly/axondeepseg/blob/master/AxonDeepSeg/morphometrics/compute_morphometrics.py
        In the future, the referenced script should be modified in order to avoid repetition.
        :param im_axon: the binary mask corresponding to axons
        :param im_myelin: the binary mask corresponding to the myelin
        :param returnCentroids:
        :return:
        """

        # Label each axon object
        im_axon_label = measure.label(im_axon)
        # Measure properties for each axon object
        axon_objects = measure.regionprops(im_axon_label)
        # Deal with myelin mask
        if im_myelin is not None:
            # sum axon and myelin masks
            im_axonmyelin = im_axon + im_myelin
            # Compute distance between each pixel and the background. Note: this distance is calculated from the im_axon,
            # note from the im_axonmyelin image, because we know that each axon object is already isolated, therefore the
            # distance metric will be more useful for the watershed algorithm below.
            distance = ndi.distance_transform_edt(im_axon)
            # local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((31, 31)), labels=axonmyelin)

            # Get axon centroid as int (not float) to be used as index
            ind_centroid = ([int(props.centroid[0]) for props in axon_objects],
                            [int(props.centroid[1]) for props in axon_objects])

            # Create an image with axon centroids, which value corresponds to the value of the axon object
            im_centroid = np.zeros_like(im_axon, dtype='uint16')
            for i in range(len(ind_centroid[0])):
                # Note: The value "i" corresponds to the label number of im_axon_label
                im_centroid[ind_centroid[0][i], ind_centroid[1][i]] = i + 1

            # markers = ndi.label(local_maxi)[0]
            # Watershed segmentation of axonmyelin using distance map
            im_axonmyelin_label = morphology.watershed(-distance, im_centroid, mask=im_axonmyelin)
            if returnCentroids is True:
                return im_axonmyelin_label, ind_centroid
            else:
                return im_axonmyelin_label

    def loadPngImageFromPath(self, imagePath, isMask=False, addToOverlayList=True, colormap='greyscale'):
        img_png = np.asarray(Image.open(imagePath).convert('LA'))

        if np.size(img_png.shape) == 3:
            img_png2D = img_png[:, :, 0]


        elif np.size(img_png.shape) == 2:
            img_png2D = img_png[:, :]

        else:
            print("Invalid image dimensions")
            return

        if isMask is True:
            img_png2D = img_png2D // 255

        img_NIfTI = nib.Nifti1Image(np.flipud(np.rot90(img_png2D, k=3, axes=(0, 1))), np.eye(4))

        # Save the NIfTI image
        outFile = imagePath[:-3] + 'nii.gz'
        nib.save(img_NIfTI, outFile, )

        img_overlay = ovLoad.loadOverlays(paths=[outFile], inmem=True, blocking=True)[0]

        if addToOverlayList is True:
            self.overlayList.append(img_overlay)
            opts = self.displayCtx.getOpts(img_overlay)
            opts.cmap = colormap

        return img_overlay

    def getCitation(self):

        return ('If you use this work in your research, please cite it as follows: \n'
                'Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). '
                'AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional '
                'neural networks. Scientific Reports, 8(1), 3816. '
                'Link to paper: https://doi.org/10.1038/s41598-018-22181-4. \n'
                'Copyright (c) 2018 NeuroPoly (Polytechnique Montreal)')

    def getLogo(self):

        ads_path = Path(os.path.abspath(AxonDeepSeg.__file__)).parents[0]

        logoFile = ads_path / 'logo_ads-alpha.png'

        png = wx.Image(str(logoFile),
                       wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        png.SetSize((png.GetWidth() // 15, png.GetHeight() // 15))
        logoImage = wx.StaticBitmap(self, -1, png, wx.DefaultPosition,
                                    (png.GetWidth(), png.GetHeight()))
        return logoImage

    @staticmethod
    def supportedViews():
        from fsleyes.views.orthopanel import OrthoPanel
        return [OrthoPanel]

    @staticmethod
    def defaultLayout():
        return {
            'location': wx.LEFT,
        }
    
