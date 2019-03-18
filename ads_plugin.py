import wx
import wx.lib.agw.hyperlink as hl

import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.actions.loadoverlay as ovLoad

import numpy as np
import nibabel as nib
from PIL import Image
import scipy.misc
import os
import json

import AxonDeepSeg
from AxonDeepSeg.apply_model import axon_segmentation

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

        separateAxons_button = wx.Button(self, label='separate axons')
        sizerV4.Add(separateAxons_button, flag=wx.SHAPED, proportion=1)

        separateAxons_button = wx.Button(self, label='fill axons')
        sizerV4.Add(separateAxons_button, flag=wx.SHAPED, proportion=1)

        self.SetSizer(sizerH)

        self.imageDirPath = None
        self.mostRecentMyelinMaskName = None
        self.mostRecentAxonMaskName = None



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





    def loadPngImageFromPath(self, imagePath, isMask=False):
        img_png = np.asarray(Image.open(imagePath).convert('LA'), dtype=np.uint8)

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
        self.overlayList.append(img_overlay)

    def getCitation(self):

        return ('If you use this work in your research, please cite it as follows: \n'
                'Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). '
                'AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional '
                'neural networks. Scientific Reports, 8(1), 3816. '
                'Link to paper: https://doi.org/10.1038/s41598-018-22181-4. \n'
                'Copyright (c) 2018 NeuroPoly (Polytechnique Montreal)')

    def getLogo(self):

        path = os.path.join(os.getcwd(), '..', '..', '..', '..')

        logoFile = path + '/ADS_logo.png'
        png = wx.Image(logoFile,
                       wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        png.SetSize((png.GetWidth()//10, png.GetHeight()//10))
        logoImage = wx.StaticBitmap(self, -1, png, wx.DefaultPosition,
                                    (png.GetWidth(), png.GetHeight()))
        return logoImage


    @staticmethod
    def supportedViews():
        from fsleyes.views.orthopanel import OrthoPanel
        return [OrthoPanel]
