import numpy as np
import imageio
import AxonDeepSeg.ads_utils

class SimulateAxons:

    def __init__(self, image_dims=[1000,1000]):
        self.width = image_dims[0]
        self.height = image_dims[1]
        self.origin = [self.width/2, self.height/2]

        # Initialize image
        self.image = np.zeros([self.width, self.height], dtype=np.uint8)

    def generate_axon(self, axon_radius, gratio = 0.7, center=None):
        if center is None:
            center=self.origin

        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        axon = (x[np.newaxis,:]-center[0])**2 + (y[:,np.newaxis]-center[1])**2 < axon_radius**2
        
        myelin_outer = (x[np.newaxis,:]-center[0])**2 + (y[:,np.newaxis]-center[1])**2 < (axon_radius/gratio)**2
        myelin = (myelin_outer ^ axon)
        
        self.image[axon & (self.image==0)]=255
        self.image[myelin & (self.image==0)]=127

    def save(self, filename="SimulatedAxons.png"):
        imageio.imwrite(filename, self.image)

    def reset(self):
        self.image = np.zeros([self.width, self.height], dtype=np.uint8)
