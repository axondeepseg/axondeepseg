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

    def __init__(self, *args, **kwargs):
        ctrlpanel.ControlPanel.__init__(self, *args, **kwargs)

        # Adding sizers to the control panel
        sizerH = wx.BoxSizer(wx.HORIZONTAL)
        sizerV1 = wx.BoxSizer(wx.VERTICAL)
        sizerV2 = wx.BoxSizer(wx.VERTICAL)
        sizerV3 = wx.BoxSizer(wx.VERTICAL)
        sizerV4 = wx.BoxSizer(wx.VERTICAL)

        sizerH.Add(sizerV1, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=10)
        sizerH.Add(sizerV2, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=10)
        sizerH.Add(sizerV3, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=10)
        sizerH.Add(sizerV4, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=10)

        # Adding widgets to the control panel

        ADS_logo = self.getLogo()
        sizerV1.Add(ADS_logo, flag=wx.SHAPED, proportion=1)


        citationBox = wx.TextCtrl(self, value=self.getCitation(), size=(100, 50), style=wx.TE_MULTILINE)
        sizerV1.Add(citationBox, flag=wx.SHAPED, proportion=1)


        loadPng_button = wx.Button(self, label="Load PNG file")
        loadPng_button.Bind(wx.EVT_BUTTON, self.onLoadPngButton)
        sizerV2.Add(loadPng_button, flag=wx.SHAPED, proportion=1)



        self.modelChoices = ['SEM', 'TEM', 'other']
        self.modelCombobox = wx.ComboBox(self, choices=self.modelChoices, size=(100, 20), value='Select the modality')
        sizerV2.Add(self.modelCombobox, flag=wx.SHAPED, proportion=1)

        applyModel_button = wx.Button(self, label='Apply ADS prediction model')
        applyModel_button.Bind(wx.EVT_BUTTON, self.onApplyModel_button)
        sizerV2.Add(applyModel_button, flag=wx.SHAPED, proportion=1)

        saveSegmentation_button = wx.Button(self, label="Save segmentation")
        saveSegmentation_button.Bind(wx.EVT_BUTTON, self.onSaveSegmentation_button)
        sizerV3.Add(saveSegmentation_button, flag=wx.SHAPED, proportion=1)

        runWatershed_button = wx.Button(self, label='Run Watershed')
        runWatershed_button.Bind(wx.EVT_BUTTON, self.onRunWatershed_button)
        sizerV4.Add(runWatershed_button, flag=wx.SHAPED, proportion=1)

        fillAxons_button = wx.Button(self, label='Fill axons')
        fillAxons_button.Bind(wx.EVT_BUTTON, self.onFillAxons_button)
        sizerV4.Add(fillAxons_button, flag=wx.SHAPED, proportion=1)

        self.SetSizer(sizerH)

        self.imageDirPath = None
        self.mostRecentMyelinMaskName = None
        self.mostRecentAxonMaskName = None
        self.mostRecentWatershedMaskName = None



    def onLoadPngButton(self, event):
        # Ask the user which file he wants to convert
        with wx.FileDialog(self, "select PNG file", wildcard="PNG files (*.png)|*.png",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            inFile = fileDialog.GetPath()


        if inFile[-4:] != '.png':
            print('This is not a png file')
            return

        # Store the directory path and image name for later use
        self.imageDirPath = os.path.dirname(inFile)
        self.pngImageName = inFile[self.imageDirPath.__len__()+1:]

        self.loadPngImageFromPath(inFile)



    def onApplyModel_button(self, event):


        if self.imageDirPath is None:
            print('Please load a PNG file')
            return

        selectedModel = self.modelCombobox.GetStringSelection()

        if selectedModel == 'other':
            # Ask the user where the model is located
            with wx.DirDialog(self, "select the directory in which the model is locatted", defaultPath="",
                              style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as fileDialog:

                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return

            modelPath = fileDialog.GetPath()

        elif (selectedModel == 'SEM') or (selectedModel == 'TEM'):
            modelPath = os.path.dirname(AxonDeepSeg.__file__)
            modelPath = os.path.join(modelPath, 'models', 'default_' + selectedModel + '_model_v1')

        else:
            print('Please select a model')
            return


        model_configfile = os.path.join(modelPath, 'config_network.json')
        with open(model_configfile, 'r') as fd:
            config_network = json.loads(fd.read())

        prediction = axon_segmentation([self.imageDirPath], [self.pngImageName], modelPath,
                                       config_network, verbosity_level=3)


        axonMaskPath = self.imageDirPath + '/AxonDeepSeg_seg-axon.png'
        myelinMaskPath = self.imageDirPath + '/AxonDeepSeg_seg-myelin.png'

        self.loadPngImageFromPath(axonMaskPath, isMask=True)
        self.loadPngImageFromPath(myelinMaskPath, isMask=True)

        self.mostRecentAxonMaskName = 'AxonDeepSeg_seg-axon'
        self.mostRecentMyelinMaskName = 'AxonDeepSeg_seg-myelin'

    def onSaveSegmentation_button(self, event):

        # Check if a mask was loaded
        if (self.mostRecentMyelinMaskName is None) or (self.mostRecentAxonMaskName is None):
            'Masks not found'
            return

        # Ask the user where to save the segmentation
        with wx.DirDialog(self, "select the directory in which the segmentation will be save", defaultPath="",
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

        saveDir = fileDialog.GetPath()

        # Find most recent segmentation masks overlays within the overlay list
        for anImage in self.overlayList:
            if anImage.name == self.mostRecentMyelinMaskName:
                myelinArray = np.array(anImage[:, :, 0]*255, copy=True, dtype=np.uint8)
                myelinArray = np.flipud(np.rot90(myelinArray,k=3, axes=(0, 1)))
            if anImage.name == self.mostRecentAxonMaskName:
                axonArray = np.array(anImage[:, :, 0]*255, copy=True, dtype=np.uint8)
                axonArray = np.flipud(np.rot90(axonArray,k=3 ,axes=(0, 1)))

        # Save the arrays as PNG files
        myelinAndAxonImage = Image.fromarray((myelinArray//2 + axonArray).astype(np.uint8))
        myelinAndAxonImage.save(saveDir + '/ADS_seg.png')
        myelinImage = Image.fromarray(myelinArray)
        myelinImage.save(saveDir + '/ADS_seg-myelin.png')
        axonImage = Image.fromarray(axonArray)
        axonImage.save(saveDir + '/ADS_seg-axon.png')


    def onRunWatershed_button(self, event):

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


            self.mostRecentWatershedMaskName = 'watershed_mask'

        elif watershedMaskOverlay is not None:
            # Update the current watershed mask
            watershedMaskOverlay[:, :, 0] = watershedData

            # This is the only way I could find to update an overlay once its data has been modified.
            # It is not very elegant. I should contact the FSL developers and ask them if there's another way to do
            # this.
            self.overlayList.remove(watershedMaskOverlay)
            self.overlayList.append(watershedMaskOverlay)

    def onFillAxons_button(self, event):
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


        # # Crate an RGB array for the myelin image. The floodfill only uses RBG images.
        # arrayShape = [myelinArray.shape[0], myelinArray.shape[1]]
        # myelinRGBArray = np.zeros(arrayShape, dtype='3uint8')


        # Create an image with the myelinMask and floodfill at the coordinates of the centroids
        myelinImage = Image.fromarray(myelinArray*255)
        myelinImage = ImageOps.colorize(myelinImage, (0, 0, 0, 255), (255, 255, 255, 255))
        for i in range(len(centroidIndexMap[0])):
            ImageDraw.floodfill(myelinImage, xy=(centroidIndexMap[1][i], centroidIndexMap[0][i]),
                                value=(127, 127, 127, 255))

        # Extract the axonArray and update the axon mask overlay
        axonExtractedArray = np.array(myelinImage.convert('LA'))
        axonExtractedArray = axonExtractedArray[:, :, 0]
        axonExtractedArray = np.equal(axonExtractedArray, 127*np.ones_like(axonExtractedArray))
        axonExtractedArray = axonExtractedArray.astype(np.uint8)


        axonMaskOverlay[:, :, 0] = axonExtractedArray
        self.overlayList.remove(axonMaskOverlay)
        self.overlayList.append(axonMaskOverlay)


    def getMyelinCentroids(self, im_myelin):
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
        :param im_axon:
        :param im_myelin:
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



    def loadPngImageFromPath(self, imagePath, isMask=False, addToOverlayList=True):
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

        return img_overlay

    def getCitation(self):

        return ('If you use this work in your research, please cite it as follows: \n'
                'Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). '
                'AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional '
                'neural networks. Scientific Reports, 8(1), 3816. '
                'Link to paper: https://doi.org/10.1038/s41598-018-22181-4. \n'
                'Copyright (c) 2018 NeuroPoly (Polytechnique Montreal)')

    def getLogo(self):

        ads_path = Path(os.path.abspath(AxonDeepSeg.__file__)).parents[1]

        logoFile = ads_path / 'docs' / 'source' / '_static' / 'logo_ads-alpha.png'

        png = wx.Image(str(logoFile),
                       wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        png.SetSize((png.GetWidth()//10, png.GetHeight()//10))
        logoImage = wx.StaticBitmap(self, -1, png, wx.DefaultPosition,
                                    (png.GetWidth(), png.GetHeight()))
        return logoImage


    @staticmethod
    def supportedViews():
        from fsleyes.views.orthopanel import OrthoPanel
        return [OrthoPanel]

