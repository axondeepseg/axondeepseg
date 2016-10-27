from skimage import exposure
from skimage.transform import rescale
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imread
from sklearn import preprocessing
import cv2
from skimage.transform import rotate
import numpy as np
import random
import os

#######################################################################################################################
#                                                Data Augmentation                                                    #
#######################################################################################################################

def extract_patch(img,mask,size):
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


def flipped(patch):
    """
    :param patch: [image,mask]
    :return: random vertical and horizontal flipped [image,mask]
    """
    s = np.random.binomial(1, 0.5, 1)
    image = patch[0]
    gt = patch[1]
    if s == 1 :
        image, gt = [np.fliplr(image), np.fliplr(gt)]
    s = np.random.binomial(1, 0.5, 1)
    if s == 1:
        image, gt=[np.flipud(image), np.flipud(gt)]
    return [image,gt]


def elastic_transform(image, gt, alpha, sigma, random_state=None):
    """
    :param image: image
    :param gt: ground truth
    :param alpha: deformation coefficient (high alpha -> strong deformation)
    :param sigma: std of the gaussian filter. (high sigma -> smooth deformation)
    :param random_state:
    :return: deformation of the pair [image,mask]
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    d = 4
    sub_shape = (shape[0]/d, shape[0]/d)

    deformations_x = random_state.rand(*sub_shape) * 2 - 1
    deformations_y = random_state.rand(*sub_shape) * 2 - 1

    deformations_x = np.repeat(np.repeat(deformations_x, d, axis=1), d, axis = 0)
    deformations_y = np.repeat(np.repeat(deformations_y, d, axis=1), d, axis = 0)

    dx = gaussian_filter(deformations_x, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(deformations_y, sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    elastic_image = map_coordinates(image, indices, order=1).reshape(shape)
    elastic_gt = map_coordinates(gt, indices, order=1).reshape(shape)
    elastic_gt = preprocessing.binarize(np.array(elastic_gt), threshold=0.5)

    return [elastic_image, elastic_gt]


def elastic(patch):
    """
    :param patch: [image,mask]
    :return: random deformation of the pair [image,mask]
    """
    alpha = random.choice([1,2,3,4,5,6,7,8,9])
    patch_deformed = elastic_transform(patch[0],patch[1], alpha = alpha, sigma = 4)
    return patch_deformed


def shifting(patch):
    """
    :param patch: [image,mask]
    :return: random shifting of the pair [image,mask]
    """
    size_shift = 10
    img = np.pad(patch[0],size_shift, mode = "reflect")
    mask = np.pad(patch[1],size_shift, mode = "reflect")
    begin_h = np.random.randint(2*size_shift-1)
    begin_w = np.random.randint(2*size_shift-1)
    shifted_image = img[begin_h:,begin_w:]
    shifted_mask = mask[begin_h:,begin_w:]

    return [shifted_image,shifted_mask]



def resc(patch):
    """
    :param patch:  [image,mask]
    :return: random rescaling of the pair [image,mask]

    --- Rescaling reinforces axons size diversity ---
    """

    s = random.choice([0.5, 0.75, 1.0, 1.5, 2.0])
    data_rescale=[]
    for scale in s:

        image_rescale = rescale(patch[0], scale)
        mask_rescale = rescale(patch[1], scale)
        s_r = mask_rescale.shape[0]
        q_h, r_h = divmod(256-s_r,2)

        if q_h > 0 :
            image_rescale = np.pad(image_rescale,(q_h, q_h+r_h), mode = "reflect")
            mask_rescale = np.pad(mask_rescale,(q_h, q_h+r_h), mode = "reflect")
        else :
            patches = extract_patch(image_rescale,mask_rescale, 256)
            i = np.random.randint(len(patches), size=1)
            image_rescale,mask_rescale = patches[i]

        mask_rescale = preprocessing.binarize(np.array(mask_rescale), threshold=0.001)
        data_rescale = [image_rescale, mask_rescale]

    return data_rescale

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def random_rotation(patch):
    """
    :param patch: [image, mask]
    :return: random rotation of the pair [image,mask]
    """

    img = np.pad(patch[0],180,mode = "reflect")
    mask = np.pad(patch[1],180,mode = "reflect")

    angle = np.random.uniform(5, 89, 1)

    image_size = (img.shape[1], img.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    #image_rotated = rotate(img, angle, resize=True, center=image_center, preserve_range=True)
    #gt_rotated = rotate(mask, angle, resize=True, center=image_center, preserve_range=True)
    #gt_rotated = (preprocessing.binarize(gt_rotated, threshold=0.5)).astype(int)

    image_rotated = rotate_image(img, angle)
    gt_rotated = rotate_image(mask, angle)

    s_p = image_rotated.shape[0]
    center = int(float(s_p)/2)

    image_rotated_cropped = image_rotated[center-128:center+128, center-128:center+128]
    gt_rotated_cropped = gt_rotated[center-128:center+128, center-128:center+128]

    return [image_rotated_cropped, gt_rotated_cropped]


def augmentation(patch):
    """
    :param patch: [image,mask]
    :param size: application of the random transformations to the pair [image,mask]
    :return: application of the random transformations to the pair [image,mask]
    """
    patch = shifting(patch)
    patch = random_rotation(patch)
    patch = elastic(patch)
    patch = flipped(patch)
    return patch


#######################################################################################################################
#                                             Feeding data for the network                                            #
#######################################################################################################################
class input_data:
    """
    Data to feed the learning/testing of the CNN
    """

    def __init__(self, trainingset_path, type = 'train'):

        if type == 'train' : # Data for the train
            self.path = trainingset_path+'/Train/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        if type == 'test': # Data for the test
            self.path = trainingset_path+'/Test/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        self.size_image = 256
        self.n_labels = 2
        self.batch_start = 0

    def set_batch_start(self, start = 0):
        """
        :param start: starting indice of the data reading by the network
        :return:
        """
        self.batch_start = start


    def next_batch(self, batch_size = 1, rnd = False, augmented_data = True):
        """
        :param batch_size: size of the batch to feed the network
        :param rnd: if True, batch is randomly taken into the training set
        :param augmented_data: if True, each patch of the batch is randomly transformed as a data augmentation process
        :return: The pair [batch_x (data), batch_y (prediction)] to feed the network
        """
        batch_x = []
        for i in range(batch_size) :
            if rnd :
                indice = random.choice(range(self.set_size))
            else :
                indice = self.batch_start
                self.batch_start += 1
                if self.batch_start >= self.set_size:
                    self.batch_start= 0

            image = imread(self.path + 'image_%s.jpeg' % indice, flatten=False, mode='L')

            #-----PreProcessing --------
            image = exposure.equalize_hist(image) #histogram equalization
            image = (image - np.mean(image))/np.std(image) #data whitening
            #---------------------------
            category = preprocessing.binarize(imread(self.path + 'mask_%s.jpeg' % indice, flatten=False, mode='L'), threshold=125)

            if augmented_data :
                [image, category] = augmentation([image, category])

            batch_x.append(image)
            if i == 0:
                batch_y = category.reshape(-1,1)
            else:
                batch_y = np.concatenate((batch_y, category.reshape(-1, 1)), axis=0)
        batch_y = np.concatenate((np.invert(batch_y)/255, batch_y), axis = 1)

        return [np.asarray(batch_x), batch_y]

    def read_batch(self, batch_y, size_batch):
        images = batch_y.reshape(size_batch, self.size_image, self.size_image, self.n_labels)
        return images

