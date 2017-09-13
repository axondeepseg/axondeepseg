import random
import numpy as np


def extract_patch(patch, size):
    """
    :param img: image represented by a numpy-array
    :param mask: groundtruth of the segmentation
    :param size: size of the patches to extract
    :return: a list of pairs [patch, ground_truth] with a very low overlapping.
    """
    img = patch[0]
    mask = patch[1]
    if len(patch) == 3:
        weights = patch[2]

    h, w = img.shape

    q_h, r_h = divmod(h, size)
    q_w, r_w = divmod(w, size)

    r2_h = size-r_h
    r2_w = size-r_w

    q3_h, r3_h = divmod(r2_h,q_h)
    q3_w, r3_w = divmod(r2_w,q_w)

    dataset = []
    pos = 0
    while pos+size<=h:
        pos2 = 0
        while pos2+size<=w:
            patch_im = img[pos:pos+size, pos2:pos2+size]
            patch_gt = mask[pos:pos+size, pos2:pos2+size,:]
            if len(patch) == 3:
                patch_weights = weights[pos:pos+size, pos2:pos2+size]
                dataset.append([patch_im, patch_gt, patch_weights])
            else:
                dataset.append([patch_im, patch_gt])
            pos2 = size + pos2 - q3_w
            if pos2 + size > w :
                pos2 = pos2 - r3_w

        pos = size + pos - q3_h
        if pos + size > h:
            pos = pos - r3_h
    return dataset