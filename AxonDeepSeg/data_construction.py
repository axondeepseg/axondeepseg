import os
import shutil
from scipy.misc import imread, imsave
from sklearn import preprocessing
from skimage.transform import rescale
import random
from config import *


def extract_patch(img, mask, size):
    """
    :param img: image represented by a numpy-array
    :param mask: groundtruth of the segmentation
    :param size: size of the patches to extract
    :return: a list of pairs [patch, ground_truth] with a very low overlapping.
    """

    h, w = img.shape

    q_h, r_h = divmod(h, size)
    q_w, r_w = divmod(w, size)

    r2_h = size-r_h
    r2_w = size-r_w
    q2_h = q_h + 1
    q2_w = q_w + 1

    q3_h, r3_h = divmod(r2_h,q_h)
    q3_w, r3_w = divmod(r2_w,q_w)

    dataset = []
    pos = 0
    while pos+size<=h:
        pos2 = 0
        while pos2+size<=w:
            patch = img[pos:pos+size, pos2:pos2+size]
            patch_gt = mask[pos:pos+size, pos2:pos2+size]
            dataset.append([patch,patch_gt])
            pos2 = size + pos2 - q3_w
            if pos2 + size > w :
                pos2 = pos2 - r3_w

        pos = size + pos - q3_h
        if pos + size > h:
            pos = pos - r3_h
    return dataset


def build_data(path_data, trainingset_path, trainRatio = 0.80):
    """
    :param path_data: folder including all images used for the training. Each image is represented by a a folder
    including image.jpg and mask.jpg (ground truth)
    :param trainingset_path: path of the resulting trainingset
    :param trainRatio: ratio of the train over the test. (High ratio : good learning but poor estimation of the performance)
    :return: no return

    Every 256 by 256 patches are extracted from the images with a very low overlapping.
    They are regrouped by category folder : \Train and \Test.
    Each data is represented by the patch, image_i.jpg, and its groundtruth, classes_i.jpg
    """

    i = 0
    for root in os.listdir(path_data)[:]:

        if '.DS_Store' not in root :
            subpath_data = os.path.join(path_data, root)

            file = open(subpath_data+'/pixel_size_in_micrometer.txt', 'r')
            pixel_size = float(file.read())
            rescale_coeff = pixel_size/general_pixel_size

            for data in os.listdir(subpath_data):
                if 'image' in data:
                    img = imread(os.path.join(subpath_data, data), flatten=False, mode='L')
                    img = (rescale(img, rescale_coeff)*256).astype(int)
                elif 'mask' in data:
                    mask_init = imread(os.path.join(subpath_data, data), flatten=False, mode='L')
                    mask_rescaled = (rescale(mask_init, rescale_coeff)*256).astype(int)
                    mask = preprocessing.binarize(mask_rescaled, threshold=125)

            if i ==0:
                patches = extract_patch(img, mask, 256)
            else:
                patches += extract_patch(img, mask, 256)
            i+=1

    testRatio = 1-trainRatio
    size_test = int(testRatio*len(patches))

    random.shuffle(patches)
    patches_train = patches[:-size_test]
    patches_test = patches[-size_test:]


    if not os.path.exists(trainingset_path):
        os.makedirs(trainingset_path)

    folder_train = trainingset_path+'/Train'

    if os.path.exists(folder_train):
        shutil.rmtree(folder_train)
    if not os.path.exists(folder_train):
        os.makedirs(folder_train)

    folder_test = trainingset_path+'/Test'

    if os.path.exists(folder_test):
        shutil.rmtree(folder_test)
    if not os.path.exists(folder_test):
        os.makedirs(folder_test)


    j = 0
    for patch in patches_train:
        imsave(folder_train+'/image_%s.jpeg'%j, patch[0],'jpeg')
        imsave(folder_train+'/classes_%s.jpeg'%j, patch[1].astype(int),'jpeg')
        j+=1

    k=0
    for patch in patches_test:
        imsave(folder_test+'/image_%s.jpeg'%k, patch[0],'jpeg')
        imsave(folder_test+'/classes_%s.jpeg'%k, patch[1].astype(int),'jpeg')
        k+=1



