from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2




def adjustData(image, mask):
    
    #print("Initial mask shape{}".format(mask.shape))
    mask = np.squeeze(mask)
    
    
    mask = descritize_mask(mask)
    #print("final mask shape{}".format(mask.shape))
    
     ## Normalizaing 
    image = image/255.0
    mask = mask/255.0
    
    return (image,mask)



def labellize_mask_2d(patch, thresh_indices=[0, 0.2, 0.8]):
    '''
    Process a patch with 8 bit pixels ([0-255]) so that the pixels between two threshold values are set to the closest threshold, effectively
    enabling the creation of a mask with as many different values as there are thresholds.

    Returns mask in [0-1] domain
    '''
    mask = np.zeros_like(patch)
    for indice in range(len(thresh_indices)-1):

        thresh_inf_8bit = 255*thresh_indices[indice]
        thresh_sup_8bit = 255*thresh_indices[indice+1]
        
        idx = np.where((patch >= thresh_inf_8bit) & (patch < thresh_sup_8bit)) # returns (x, y) of the corresponding indices
        mask[idx] = np.mean([thresh_inf_8bit/255,thresh_sup_8bit/255])

    mask[(patch >= 255*thresh_indices[-1])] = 1

    return patch


def descritize_mask(mask, thresh_indices = [0, 0.2, 0.8]):
# Discretization of the mask
        mask = labellize_mask_2d(mask, thresh_indices) # mask intensity float between 0-1

        # Working out the real mask (sparse cube with n depth layer, one for each class)
        n = len(thresh_indices) # number of classes
        thresh_indices = [255*x for x in thresh_indices]
        real_mask = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], n])
       
        for class_ in range(n-1):
            real_mask[:,:,:,class_] = (mask[:,:,:] >= thresh_indices[class_]) * (mask[:,:,:] <  thresh_indices[class_+1])
        real_mask[:,:,:,-1] = (mask[:,:,:] >= thresh_indices[-1])
        real_mask = real_mask.astype(np.uint8)

        return real_mask
    
    
    

def Generator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        interpolation = "bilinear")
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        interpolation = "bilinear")
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)

        
        
def testGenerator(test_path,image = 16,target_size = (512,512),flag_multi_class = False,as_gray = False):
    for i in range(num_image):
        path = os.path.join(test_path,"image_%d.png"%i)
        img = cv2.imread(path)
        img = cv2.resize(img, target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img
        


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)     
  

        
