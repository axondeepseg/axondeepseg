# Gathers functions used for patch management, including preprocessing.
import numpy as np
import AxonDeepSeg.ads_utils


def im2patches_overlap(img, overlap_value=25, scw=512):

    '''
    Convert an image into patches.
    :param img: the image to convert.
    :param overlap_value: Int, the number of pixels to use when overlapping the predictions.
    :param scw: Int, input size.
    :return: the original image, a list of patches, and their positions.
    '''

    # First we crop the image to get the context
    cropped = img[overlap_value:-overlap_value, overlap_value:-overlap_value]

    # Then we create patches using the prediction window
    spw = scw - 2 * overlap_value  # size prediction windows

    qh, rh = divmod(cropped.shape[0], spw)
    qw, rw = divmod(cropped.shape[1], spw)

    # Creating positions of prediction windows
    L_h = [spw * e for e in range(qh)]
    L_w = [spw * e for e in range(qw)]

    # Then if there is a remainder we take the last positions (overlap on the last predictions)
    if rh != 0:
        L_h.append(cropped.shape[0] - spw)
    if rw != 0:
        L_w.append(cropped.shape[1] - spw)

    xx, yy = np.meshgrid(L_h, L_w)
    P = [np.ravel(xx), np.ravel(yy)]
    L_pos = [[P[0][i], P[1][i]] for i in range(len(P[0]))]

    # These positions are also the positions of the context windows in the base image coordinates !
    L_patches = []
    for e in L_pos:
        patch = img[e[0]:e[0] + scw, e[1]:e[1] + scw]
        L_patches.append(patch)

    return [img, L_patches, L_pos]


def patches2im_overlap(L_patches, L_pos, overlap_value=25, scw=512):

    '''
    Stitches patches together to form an image.
    :param L_patches: List of segmented patches.
    :param L_pos: List of positions of the patches in the image to form.
    :param overlap_value: Int, number of pixels to overlap.
    :param scw: Int, patch size.
    :return: Stitched segmented image.
    '''

    spw = scw - 2 * overlap_value
    # L_pred = [e[cropped_value:-cropped_value,cropped_value:-cropped_value] for e in L_patches]
    # First : extraction of the predictions
    h_l, w_l = np.max(np.stack(L_pos), axis=0)
    L_pred = []
    new_img = np.zeros((h_l + scw, w_l + scw))

    for i, e in enumerate(L_patches):
        if L_pos[i][0] == 0:
            if L_pos[i][1] == 0:
                new_img[0:overlap_value, 0:overlap_value] = e[0:overlap_value, 0:overlap_value]
                new_img[overlap_value:scw - overlap_value, 0:overlap_value] = e[overlap_value:-overlap_value,
                                                                              0:overlap_value]
                new_img[0:overlap_value, overlap_value:scw - overlap_value] = e[0:overlap_value,
                                                                              overlap_value:-overlap_value]
            else:
                if L_pos[i][1] == w_l:
                    new_img[0:overlap_value, -overlap_value:] = e[0:overlap_value, -overlap_value:]
                new_img[0:overlap_value, L_pos[i][1] + overlap_value:L_pos[i][1] + scw - overlap_value] = e[
                                                                                                          0:overlap_value,
                                                                                                          overlap_value:-overlap_value]

        if L_pos[i][1] == 0:
            if L_pos[i][0] != 0:
                new_img[L_pos[i][0] + overlap_value:L_pos[i][0] + scw - overlap_value, 0:overlap_value] = e[
                                                                                                          overlap_value:-overlap_value,
                                                                                                          0:overlap_value]

        if L_pos[i][0] == h_l:
            if L_pos[i][1] == w_l:
                new_img[-overlap_value:, -overlap_value:] = e[-overlap_value:, -overlap_value:]
                new_img[h_l + overlap_value:-overlap_value, -overlap_value:] = e[overlap_value:-overlap_value,
                                                                               -overlap_value:]
                new_img[-overlap_value:, w_l + overlap_value:-overlap_value] = e[-overlap_value:,
                                                                               overlap_value:-overlap_value]
            else:
                if L_pos[i][1] == 0:
                    new_img[-overlap_value:, 0:overlap_value] = e[-overlap_value:, 0:overlap_value]

                new_img[-overlap_value:, L_pos[i][1] + overlap_value:L_pos[i][1] + scw - overlap_value] = e[
                                                                                                          -overlap_value:,
                                                                                                          overlap_value:-overlap_value]
        if L_pos[i][1] == w_l:
            if L_pos[i][1] != h_l:
                new_img[L_pos[i][0] + overlap_value:L_pos[i][0] + scw - overlap_value, -overlap_value:] = e[
                                                                                                          overlap_value:-overlap_value,
                                                                                                          -overlap_value:]

    L_pred = [e[overlap_value:-overlap_value, overlap_value:-overlap_value] for e in L_patches]
    L_pos_corr = [[e[0] + overlap_value, e[1] + overlap_value] for e in L_pos]
    for i, e in enumerate(L_pos_corr):
        new_img[e[0]:e[0] + spw, e[1]:e[1] + spw] = L_pred[i]

    return new_img
