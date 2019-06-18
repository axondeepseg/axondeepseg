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

import tempfile


class ADScontrol(ctrlpanel.ControlPanel):
    """
    This class is the object corresponding to the AxonDeepSeg control panel.
    """

    def __init__(self, ortho, *args, **kwargs):
        """
        This function initializes the control panel. It generates the widgets and adds them to the panel. It also sets
        the initial position of the panel to the left
        :param ortho: This is used to access the ortho ops in order to turn off the X and Y canvas as well as the cursor
        """
        ctrlpanel.ControlPanel.__init__(self, ortho, *args, **kwargs)

        # Add a sizer to the control panel
        # This sizer will contain the buttons
        sizer_h = wx.BoxSizer(wx.VERTICAL)


        # Add the logo to the control panel
        ADS_logo = self.getLogo()
        sizer_h.Add(ADS_logo, flag=wx.SHAPED, proportion=1)

        # Add the citation to the control panel
        citation_box = wx.TextCtrl(self, value=self.getCitation(), size=(100, 50), style=wx.TE_MULTILINE)
        sizer_h.Add(citation_box, flag=wx.SHAPED, proportion=1)

        # Add the image loading button
        load_png_button = wx.Button(self, label="Load PNG or TIF file")
        load_png_button.Bind(wx.EVT_BUTTON, self.onLoadPngButton)
        load_png_button.SetToolTip(wx.ToolTip("Loads a .png or .tif file into FSLeyes"))
        sizer_h.Add(load_png_button, flag=wx.SHAPED, proportion=1)

        # Add the mask loading button
        load_mask_button = wx.Button(self, label="Load existing mask")
        load_mask_button.Bind(wx.EVT_BUTTON, self.onLoadMaskButton)
        load_mask_button.SetToolTip(wx.ToolTip("Loads an existing axon or myelin mask into FSLeyes"))
        sizer_h.Add(load_mask_button, flag=wx.SHAPED, proportion=1)

        # Add the save Segmentation button
        save_segmentation_button = wx.Button(self, label="Save segmentation")
        save_segmentation_button.Bind(wx.EVT_BUTTON, self.onSaveSegmentation_button)
        save_segmentation_button.SetToolTip(wx.ToolTip("Saves the axon and myelin masks in the selected folder"))
        sizer_h.Add(save_segmentation_button, flag=wx.SHAPED, proportion=1)

        # Add the model choice combobox
        self.model_choices = ['SEM', 'TEM', 'other']
        self.model_combobox = wx.ComboBox(self, choices=self.model_choices, size=(100, 20), value='Select the modality')
        self.model_combobox.SetToolTip(wx.ToolTip("Select the modality used to acquire the image"))
        sizer_h.Add(self.model_combobox, flag=wx.SHAPED, proportion=1)

        # Add the button that applies the prediction model
        apply_model_button = wx.Button(self, label='Apply ADS prediction model')
        apply_model_button.Bind(wx.EVT_BUTTON, self.onApplyModel_button)
        apply_model_button.SetToolTip(wx.ToolTip("Applies the prediction model and displays the masks"))
        sizer_h.Add(apply_model_button, flag=wx.SHAPED, proportion=1)

        # Add the button that runs the watershed algorithm
        run_watershed_button = wx.Button(self, label='Run Watershed')
        run_watershed_button.Bind(wx.EVT_BUTTON, self.onRunWatershed_button)
        run_watershed_button.SetToolTip(wx.ToolTip("Uses a watershed algorithm to find the different axon+myelin"
                                                  "objects. This is used to see if where are connections"
                                                  " between two axon+myelin objects."))
        sizer_h.Add(run_watershed_button, flag=wx.SHAPED, proportion=1)

        # Add the fill axon tool
        fill_axons_button = wx.Button(self, label='Fill axons')
        fill_axons_button.Bind(wx.EVT_BUTTON, self.onFillAxons_button)
        fill_axons_button.SetToolTip(wx.ToolTip("Automatically fills the axons inside myelin objects."
                                               " THE MYELIN OBJECTS NEED TO BE CLOSED AND SEPARATED FROM EACH "
                                               "OTHER (THEY MUST NOT TOUCH) FOR THIS TOOL TO WORK CORRECTLY."))
        sizer_h.Add(fill_axons_button, flag=wx.SHAPED, proportion=1)

        # Set the sizer of the control panel
        self.SetSizer(sizer_h)

        # Initialize the variables that are used to track the active image
        self.png_image_name = []
        self.image_dir_path = []
        self.most_recent_watershed_mask_name = None

        # Toggle off the X and Y canvas
        oopts = ortho.sceneOpts
        oopts.showXCanvas = False
        oopts.showYCanvas = False

        # Toggle off the cursor
        oopts.showCursor = False

        # Create a temporary directory that will hold the NIfTI files
        self.ads_temp_dir = tempfile.TemporaryDirectory()

    def onLoadPngButton(self, event):
        """
        This function is called when the user presses on the Load Png button. It allows the user to select a PNG or TIF
        image, convert it into a NIfTI and load it into FSLeyes.
        """
        # Ask the user which file he wants to convert
        with wx.FileDialog(self, "select Image file",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:

            if file_dialog.ShowModal() == wx.ID_CANCEL:  # The user cancelled the operation
                return

            in_file = file_dialog.GetPath()

        # Check if the image format is valid
        if (in_file[-4:] != '.png') and (in_file[-4:] != '.tif'):
            self.show_message('Invalid file extension')
            return

        # Store the directory path and image name for later use in the application of the prediction model
        self.image_dir_path.append(os.path.dirname(in_file))
        self.png_image_name.append(in_file[os.path.dirname(in_file).__len__() + 1:])

        # Call the function that convert and loads the png or tif image
        self.loadPngImageFromPath(in_file)

    def onLoadMaskButton(self, event):
        """
        This function is called when the user presses on the loadMask button. It allows the user to select an existing
        PNG mask, convert it into a NIfTI and load it into FSLeyes.
        """
        # Ask the user to select the mask image
        with wx.FileDialog(self, "select mask .png file",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:

            if file_dialog.ShowModal() == wx.ID_CANCEL:  # The user cancelled the operation
                return

            in_file = file_dialog.GetPath()

        # Check if the image format is valid
        if (in_file[-4:] != '.png'):
            self.show_message('Invalid file extension')
            return
        # Load the mask into FSLeyes
        if ('axon' in in_file) :
            self.loadPngImageFromPath(in_file, isMask=True, colormap='blue')
        elif ('myelin' in in_file) or ('Myelin' in in_file):
            self.loadPngImageFromPath(in_file, isMask=True, colormap='red')
        else:
            self.loadPngImageFromPath(in_file, isMask=True)

    def onApplyModel_button(self, event):
        """
        This function is called when the user presses on the ApplyModel button. It is used to apply the prediction model
        selected in the combobox. The segmentation masks are then loaded into FSLeyes
        """

        # Get the image name and directory
        image_overlay = self.getVisibleImageOverlay()
        if self.getVisibleImageOverlay() is None:
            return

        n_loaded_images = self.png_image_name.__len__()
        image_name = None
        image_directory = None
        for i in range(n_loaded_images):
            if image_overlay.name == (self.png_image_name[i])[:-4]:
                image_name = self.png_image_name[i]
                image_directory = self.image_dir_path[i]

        if (image_name is None) or (image_directory is None):
            self.show_message("Couldn't find the path to the loaded image")
            return

        # Get the selected model
        selected_model = self.model_combobox.GetStringSelection()


        # Get the path of the selected model
        if selected_model == 'other':
            # Ask the user where the model is located
            with wx.DirDialog(self, "select the directory in which the model is located", defaultPath="",
                              style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as file_dialog:

                if file_dialog.ShowModal() == wx.ID_CANCEL:  # The user cancelled the operation
                    return

            model_path = file_dialog.GetPath()


        elif (selected_model == 'SEM') or (selected_model == 'TEM'):
            model_path = os.path.dirname(AxonDeepSeg.__file__)
            model_path = os.path.join(model_path, 'models', 'default_' + selected_model + '_model_v1')

        else:
            self.show_message('Please select a model')
            return

        # Check if the pixel size txt file exist in the imageDirPath
        pixel_size_exists = os.path.isfile(image_directory + '/pixel_size_in_micrometer.txt')

        # if it doesn't exist, ask the user to input the pixel size and create the .txt file
        if pixel_size_exists is False:
            with wx.TextEntryDialog(self, "Enter the pixel size in micrometer", value="0.07") as text_entry:
                if text_entry.ShowModal() == wx.ID_CANCEL:
                    return

                pixel_size_str = text_entry.GetValue()
            text_file = open(image_directory + '/pixel_size_in_micrometer.txt', 'w')
            text_file.write(pixel_size_str)
            text_file.close()

        # Load model configs and apply prediction
        model_configfile = os.path.join(model_path, 'config_network.json')
        with open(model_configfile, 'r') as fd:
            config_network = json.loads(fd.read())

        prediction = axon_segmentation([image_directory], [image_name], model_path,
                                       config_network, verbosity_level=3)
        # The axon_segmentation function creates the segmentation masks and stores them as PNG files in the same folder
        # as the original image file.

        # Load the axon and myelin masks into FSLeyes
        axon_mask_path = image_directory + '/AxonDeepSeg_seg-axon.png'
        myelin_mask_path = image_directory + '/AxonDeepSeg_seg-myelin.png'
        self.loadPngImageFromPath(axon_mask_path, isMask=True, colormap='blue')
        self.loadPngImageFromPath(myelin_mask_path, isMask=True, colormap='red')



    def onSaveSegmentation_button(self, event):
        """
        This function saves the active myelin and axon masks as PNG images. Three (3) images are generated in a folder
        selected by the user : one with the axon mask, one with the myelin mask and one with both.
        """

        # Find the visible myelin and axon masks
        axon_mask_overlay = self.getVisibleAxonOverlay()
        myelin_mask_overlay = self.getVisibleMyelinOverlay()

        if (axon_mask_overlay is None) or (myelin_mask_overlay is None):
            return

        # Ask the user where to save the segmentation
        with wx.DirDialog(self, "select the directory in which the segmentation will be save", defaultPath="",
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as file_dialog:

            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return

        save_dir = file_dialog.GetPath()

        # store the data of the masks in variables as numpy arrays.
        # Note: since PIL uses a different convention for the X and Y coordinates, some array manipulation has to be
        # done.

        myelin_array = np.array(myelin_mask_overlay[:, :, 0] * 255, copy=True, dtype=np.uint8)
        myelin_array = np.flipud(np.rot90(myelin_array, k=3, axes=(0, 1)))
        axon_array = np.array(axon_mask_overlay[:, :, 0] * 255, copy=True, dtype=np.uint8)
        axon_array = np.flipud(np.rot90(axon_array, k=3, axes=(0, 1)))

        # Make sure the masks have the same size
        if (myelin_array.shape != axon_array.shape):
            self.show_message('invalid visible masks dimensions')
            return

        # Save the arrays as PNG files
        myelin_and_axon_image = Image.fromarray((myelin_array // 2 + axon_array).astype(np.uint8))
        myelin_and_axon_image.save(save_dir + '/ADS_seg.png')
        myelin_image = Image.fromarray(myelin_array)
        myelin_image.save(save_dir + '/ADS_seg-myelin.png')
        axon_image = Image.fromarray(axon_array)
        axon_image.save(save_dir + '/ADS_seg-axon.png')

    def onRunWatershed_button(self, event):
        """
        This function is called then the user presses on the runWatershed button. This creates a watershed mask that is
        used to locate where are the connections between the axon-myelin objects.
        """


        # Find the visible myelin and axon masks
        axon_mask_overlay = self.getVisibleAxonOverlay()
        myelin_mask_overlay = self.getVisibleMyelinOverlay()

        if (axon_mask_overlay is None) or (myelin_mask_overlay is None):
            return

        # Extract the data from the overlays
        axon_array = axon_mask_overlay[:, :, 0]
        myelin_array = myelin_mask_overlay[:, :, 0]

        # Make sure the masks have the same size
        if (myelin_array.shape != axon_array.shape):
            self.show_message('invalid visible masks dimensions')
            return

        # If a watershed mask already exists, remove it.
        for anOverlay in self.overlayList:
            if (self.most_recent_watershed_mask_name is not None) and\
                    (anOverlay.name == self.most_recent_watershed_mask_name):
                self.overlayList.remove(anOverlay)

        # Compute the watershed mask
        watershedData = self.getWatershedSegmentation(axon_array, myelin_array)

        # Save the watershed mask as a png then load it as an overlay
        watershedImageArray = np.flipud(np.rot90(watershedData, k=3, axes=(0, 1)))
        watershedImage = Image.fromarray(watershedImageArray)
        fileName = self.ads_temp_dir.name + '/watershed_mask.png'
        watershedImage.save(fileName)
        watershedMaskOverlay = self.loadPngImageFromPath(fileName, addToOverlayList=False)
        watershedMaskOverlay[:, :, 0] = watershedData
        self.overlayList.append(watershedMaskOverlay)

        # Apply a "random" colour mapping to the watershed mask
        opts = self.displayCtx.getOpts(watershedMaskOverlay)
        opts.cmap = 'random'

        self.most_recent_watershed_mask_name = 'watershed_mask'


    def onFillAxons_button(self, event):
        """
        This function is called when the fillAxon button is pressed by the user. It uses a flood fill algorithm to fill
        the inside of the myelin objects with the axon mask
        """
        # Find the visible myelin and axon masks
        axon_mask_overlay = self.getVisibleAxonOverlay()
        myelin_mask_overlay = self.getVisibleMyelinOverlay()

        if (axon_mask_overlay is None) or (myelin_mask_overlay is None):
            return

        # Extract the data from the overlays
        axon_array = axon_mask_overlay[:, :, 0]
        myelin_array = myelin_mask_overlay[:, :, 0]

        # Make sure the masks have the same size
        if (myelin_array.shape != axon_array.shape):
            self.show_message('invalid visible masks dimensions')
            return

        # Get the centroid indexes
        centroidIndexMap = self.getMyelinCentroids(myelin_array)


        # Create an image with the myelinMask and floodfill at the coordinates of the centroids
        # Note: The floodfill algorithm only works on PNG images. Thus, the mask must be colorized before applying
        # the floodfill. Then, the array corresponding to the floodfilled color can be extracted.
        myelinImage = Image.fromarray(myelin_array * 255)
        myelinImage = ImageOps.colorize(myelinImage, (0, 0, 0, 255), (255, 255, 255, 255))
        for i in range(len(centroidIndexMap[0])):
            ImageDraw.floodfill(myelinImage, xy=(centroidIndexMap[1][i], centroidIndexMap[0][i]),
                                value=(127, 127, 127, 255))

        # Extract the axon_array and update the axon mask overlay
        axonExtractedArray = np.array(myelinImage.convert('LA'))
        axonExtractedArray = axonExtractedArray[:, :, 0]
        axonExtractedArray = np.equal(axonExtractedArray, 127 * np.ones_like(axonExtractedArray))
        axonExtractedArray = axonExtractedArray.astype(np.uint8)

        axon_mask_overlay[:, :, :] = axonExtractedArray

    def getMyelinCentroids(self, im_myelin):
        """
        This function is used to find the centroids of the myelin mask
        :param im_myelin: the binary mask corresponding to the myelin
        :type im_myelin: ndarray
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
        :type im_axon: ndarray
        :param im_myelin: the binary mask corresponding to the myelin
        :type im_myelin: ndarray
        :param returnCentroids: (optional) if this is set to true, the function will also return the centroids of the
        axon objects as a list of tuples
        :type returnCentroids: bool
        :return: the label corresponding to the axon+myelin objects
        :rtype: ndarray
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

            # Watershed segmentation of axonmyelin using distance map
            im_axonmyelin_label = morphology.watershed(-distance, im_centroid, mask=im_axonmyelin)
            if returnCentroids is True:
                return im_axonmyelin_label, ind_centroid
            else:
                return im_axonmyelin_label

    def loadPngImageFromPath(self, imagePath, isMask=False, addToOverlayList=True, colormap='greyscale'):
        """
        This function converts a 2D image into a NIfTI image and loads it as an overlay.
        The parameter addToOverlayList allows to display the overlay into FSLeyes.
        :param imagePath: The location of the image, including the name and the .extension
        :type imagePath: string
        :param isMask: (optional) Whether or not this is a segmentation mask. It will be treated as a normal
        image by default.
        :type isMask: bool
        :param addToOverlayList: (optional) Whether or not to add the image to the overlay list. If so, the image will
        be displayed in the application. This parameter is True by default.
        :type addToOverlayList: bool
        :param colormap: (optional) the colormap of image that will be displayed. This parameter is set to greyscale by
        default.
        :type colormap: string
        :return: the FSLeyes overlay corresponding to the loaded image.
        :rtype: overlay
        """

        # Open the 2D image
        img_png = np.asarray(Image.open(imagePath).convert('LA'))

        # Extract the image data as a 2D NumPy array
        if np.size(img_png.shape) == 3:
            img_png2D = img_png[:, :, 0]


        elif np.size(img_png.shape) == 2:
            img_png2D = img_png[:, :]

        else:
            self.show_message("Invalid image dimensions")
            return

        if isMask is True:
            img_png2D = img_png2D // 255  # Segmentation masks should be binary

        # Convert image data into a NIfTI image
        # Note: PIL and NiBabel use different axis conventions, so some array manipulation has to be done.
        img_NIfTI = nib.Nifti1Image(np.flipud(np.rot90(img_png2D, k=3, axes=(0, 1))), np.eye(4))

        # Save the NIfTI image in a temporary directory
        img_name = os.path.basename(imagePath)
        outFile = self.ads_temp_dir.name + '/' + img_name[:-3] + 'nii.gz'
        nib.save(img_NIfTI, outFile, )

        # Load the NIfTI image as an overlay
        img_overlay = ovLoad.loadOverlays(paths=[outFile], inmem=True, blocking=True)[0]

        # Display the overlay
        if addToOverlayList is True:
            self.overlayList.append(img_overlay)
            opts = self.displayCtx.getOpts(img_overlay)
            opts.cmap = colormap

        return img_overlay

    def getVisibleOverlays(self):
        """
        This function returns a list containing evey overlays that are visible on FSLeyes.
        :return: The list of the visible overlays
        :rtype: list
        """

        visibleOverlayList = []
        for anOverlay in self.overlayList:
            anOverlayDisplay = self.displayCtx.getDisplay(anOverlay)
            if anOverlayDisplay.enabled is True:
                visibleOverlayList.append(anOverlay)

        return visibleOverlayList

    def getVisibleImageOverlay(self):
        """
        This function is used to find the active microscopy image. This image should be visible and should NOT have the
        following keywords in its name : axon, myelin, Myelin, watershed, Watershed.
        :return: The visible microscopy image
        :rtype: overlay
        """
        visibleOverlayList = self.getVisibleOverlays()
        image_overlay = None
        nFoundOverlays = 0

        if visibleOverlayList.__len__() is 0:
            self.show_message('No overlays are displayed')
            return None

        if visibleOverlayList.__len__() is 1:
            return visibleOverlayList[0]

        for anOverlay in visibleOverlayList:
            if ('axon' not in anOverlay.name) and ('myelin' not in anOverlay.name) and\
                    ('Myelin' not in anOverlay.name) and ('watershed' not in anOverlay.name) and\
                    ('Watershed' not in anOverlay.name):
                nFoundOverlays = nFoundOverlays + 1
                image_overlay = anOverlay

        if nFoundOverlays > 1:
            self.show_message('More than one microscopy image has been found')
            return None
        if nFoundOverlays is 0:
            self.show_message('No visible microscopy image has been found')
            return None

        return image_overlay

    def getVisibleAxonOverlay(self):
        """
        This method finds the currently visible axon overlay
        :return: The visible overlay that corresponds to the axon mask
        :rtype: overlay
        """
        visibleOverlayList = self.getVisibleOverlays()
        axonOverlay = None
        nFoundOverlays = 0

        if visibleOverlayList.__len__() is 0:
            self.show_message('No overlays are displayed')
            return None

        for anOverlay in visibleOverlayList:
            if ('axon' in anOverlay.name):
                nFoundOverlays = nFoundOverlays + 1
                axonOverlay = anOverlay

        if nFoundOverlays > 1:
            self.show_message('More than one axon mask has been found')
            return None
        if nFoundOverlays is 0:
            self.show_message('No visible axon mask has been found')
            return None

        return axonOverlay

    def getVisibleMyelinOverlay(self):
        """
        This method finds the currently visible myelin overlay
        :return: The visible overlay that corresponds to the myelin mask
        :rtype: overlay
        """
        visibleOverlayList = self.getVisibleOverlays()
        myelinOverlay = None
        nFoundOverlays = 0

        if visibleOverlayList.__len__() is 0:
            self.show_message('No overlays are displayed')
            return None

        for anOverlay in visibleOverlayList:
            if ('myelin' in anOverlay.name) or ('Myelin' in anOverlay.name):
                nFoundOverlays = nFoundOverlays + 1
                myelinOverlay = anOverlay

        if nFoundOverlays > 1:
            self.show_message('More than one myelin mask has been found')
            return None
        if nFoundOverlays is 0:
            self.show_message('No visible myelin mask has been found')
            return None

        return myelinOverlay

    def show_message(self, message, caption='Error'):
        """
        This function is used to show a popup message on the FSLeyes interface.
        :param message: The message to be displayed.
        :type message: String
        :param caption: (Optional) The caption of the message box.
        :type caption: String
        """
        with wx.MessageDialog(self, message, caption=caption, style=wx.OK | wx.CENTRE, pos=wx.DefaultPosition) as msg:
            msg.ShowModal()

    def getCitation(self):
        """
        This function returns the AxonDeepSeg paper citation.
        :return: The AxonDeepSeg citation
        :rtype: string
        """

        return ('If you use this work in your research, please cite it as follows: \n'
                'Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). '
                'AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional '
                'neural networks. Scientific Reports, 8(1), 3816. '
                'Link to paper: https://doi.org/10.1038/s41598-018-22181-4. \n'
                'Copyright (c) 2018 NeuroPoly (Polytechnique Montreal)')

    def getLogo(self):
        """
        This function finds the AxonDeepSeg logo saved as a png image and returns it as a wx bitmap image.
        :return: The AxonDeepSeg logo
        :rtype: wx.StaticBitmap
        """

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
        """
        I am not sure what this method does.
        """
        from fsleyes.views.orthopanel import OrthoPanel
        return [OrthoPanel]

    @staticmethod
    def defaultLayout():
        """
        This method makes the control panel appear on the left of the FSLeyes window.
        """
        return {
            'location': wx.LEFT,
        }
    
