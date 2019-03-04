import wx
import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.actions.loadoverlay as ovLoad

import numpy as np
import nibabel as nib
from PIL import Image
import os
import json

# from AxonDeepSeg.apply_model import axon_segmentation

class Ads_control_v2(ctrlpanel.ControlPanel):

    def __init__(self, *args, **kwargs):
        ctrlpanel.ControlPanel.__init__(self, *args, **kwargs)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        sizer.Add(wx.StaticText(self, label='Ads_plugin_v2_controls'),
                  flag=wx.EXPAND, proportion=1)

        self.imageDirPath = None

        loadPng_button = wx.Button(self, label = "Load PNG file")
        loadPng_button.Bind(wx.EVT_BUTTON, self.onLoadPngButton)
        sizer.Add(loadPng_button, flag=wx.SHAPED, proportion=1)

        applyModel_button = wx.Button(self, label = 'Apply ADS prediction model')
        applyModel_button.Bind(wx.EVT_BUTTON, self.onApplyModel_button)
        sizer.Add(applyModel_button, flag=wx.SHAPED, proportion=1)



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

        # Convert the png image into NIfTI
        img_png = np.asarray(Image.open(inFile).convert('LA'), dtype=np.uint8)

        if np.size(img_png.shape) == 3:
            img_png2D = img_png[:,:,0]


        elif np.size(img_png.shape) == 2:
            img_png2D = img_png[:, :]

        else:
            print("Invalid image dimensions")
            return

        img_NIfTI = nib.Nifti1Image(np.flipud(np.rot90(img_png2D, k=3, axes=(0,1))), np.eye(4))


        # Save the NIfTI image
        outFile = inFile[:-3] + 'nii.gz'
        nib.save(img_NIfTI, outFile, )

        img_overlay = ovLoad.loadOverlays(paths=[outFile], inmem=True, blocking=True)[0]
        self.overlayList.append(img_overlay)

    def onApplyModel_button(self, event):

        print('work in progress')
        # Axondeepseg and FSLeyes have incompatible numpy versions

        if self.imageDirPath is None:
            print('Please load a PNG file')
            return

        # Ask the user where the model is located
        # Ask the user which file he wants to convert
        with wx.DirDialog(self, "select the directory in which the model is locatted", defaultPath="",
                           style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

        modelPath = fileDialog.GetPath()





        model_configfile = os.path.join(modelPath, 'config_network.json')
        with open(model_configfile, 'r') as fd:
            config_network = json.loads(fd.read())

        print('work in progress')









    @staticmethod
    def supportedViews():
        from fsleyes.views.orthopanel import OrthoPanel
        return [OrthoPanel]