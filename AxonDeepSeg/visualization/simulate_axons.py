import numpy as np
from scipy.misc import imrotate
import imageio
import AxonDeepSeg.ads_utils

class SimulateAxons:

    def __init__(self, image_dims=[1000,1000]):
        self.width = image_dims[0]
        self.height = image_dims[1]
        self.origin = [self.width/2, self.height/2]

        # Initialize image
        self.image = np.zeros([self.width, self.height], dtype=np.uint8)

    def generate_axon(self, axon_radius, center=None, gratio = 0.7, axon_angle = 0, plane_angle = 0, eccentricity = 0):
        if center is None:
            center=self.origin

        axon_major_axis_radius = axon_radius/np.cos(np.deg2rad(plane_angle))

        if eccentricity != 0:
            axon_minor_axis_radius = np.sqrt(axon_radius**2-(eccentricity*axon_radius)**2)
        else:
            axon_minor_axis_radius = axon_radius

        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        axon = (x[np.newaxis,:]-center[0])**2/(axon_major_axis_radius**2) + (y[:,np.newaxis]-center[1])**2/(axon_minor_axis_radius**2) < 1
        
        myelin_outer = (x[np.newaxis,:]-center[0])**2/(axon_major_axis_radius/gratio)**2 + (y[:,np.newaxis]-center[1])**2/(axon_minor_axis_radius/gratio)**2 < 1
        myelin = (myelin_outer ^ axon)
        
        # Convert to int befor rotate
        axon = np.ndarray.astype(axon, int)
        myelin = np.ndarray.astype(myelin, int)

        axon = imrotate(axon, axon_angle, 'nearest')
        myelin = imrotate(myelin, axon_angle, 'nearest')

        # Convert back to bool
        axon = np.ndarray.astype(axon, bool)
        myelin = np.ndarray.astype(myelin, bool)

        self.image[axon & (self.image==0)]=255
        self.image[myelin & (self.image==0)]=127

    def save(self, filename="SimulatedAxons.png"):
        imageio.imwrite(filename, self.image)

    def reset(self):
        self.image = np.zeros([self.width, self.height], dtype=np.uint8)
