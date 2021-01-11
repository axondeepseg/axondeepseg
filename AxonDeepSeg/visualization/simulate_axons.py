import numpy as np
from scipy.ndimage import rotate
import imageio
import AxonDeepSeg.ads_utils

class SimulateAxons:

    def __init__(self, image_dims=[1000,1000]):
        self.width = image_dims[0]
        self.height = image_dims[1]
        self.origin = [self.width/2, self.height/2]

        # Initialize image
        self.image = np.zeros([self.width, self.height], dtype=np.uint8)

    def generate_axon(
            self,
            axon_radius,
            center=None,
            gratio = 0.7,
            ellipse_mode = None,
            ellipse_ratio = 1,
            plane_angle = 0
            ):

        if center is None:
            center=self.origin

        x = np.arange(0, self.width)
        y = np.arange(0, self.height)

        if ellipse_mode is 'concentric':

            myelin_thickness = axon_radius*(1-gratio)

            axon_major_axis_radius = axon_radius*ellipse_ratio
            axon_minor_axis_radius = axon_radius

            myelin_major_axis_radius = axon_major_axis_radius + myelin_thickness
            myelin_minor_axis_radius = axon_minor_axis_radius + myelin_thickness

        else:
            myelin_thickness = axon_radius*(1-gratio)

            axon_major_axis_radius = axon_radius/np.cos(np.deg2rad(plane_angle))
            axon_minor_axis_radius = axon_radius

            myelin_major_axis_radius = axon_major_axis_radius + myelin_thickness
            myelin_minor_axis_radius = axon_minor_axis_radius + myelin_thickness

        axon = (x[np.newaxis,:]-center[0])**2/(axon_major_axis_radius**2) + (y[:,np.newaxis]-center[1])**2/(axon_minor_axis_radius**2) < 1
        myelin_outer = (x[np.newaxis,:]-center[0])**2/(myelin_major_axis_radius**2) + (y[:,np.newaxis]-center[1])**2/(myelin_minor_axis_radius**2) < 1
        
        myelin = (myelin_outer ^ axon)
            
        # Convert to int before rotate
        axon = np.ndarray.astype(axon, int)
        myelin = np.ndarray.astype(myelin, int)

        # Convert back to bool
        axon = np.ndarray.astype(axon, bool)
        myelin = np.ndarray.astype(myelin, bool)

        self.image[axon & (self.image==0)]=255
        self.image[myelin & (self.image==0)]=127

    def save(self, filename="SimulatedAxons.png"):
        imageio.imwrite(filename, self.image)

    def reset(self):
        self.image = np.zeros([self.width, self.height], dtype=np.uint8)