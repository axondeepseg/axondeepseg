from skimage import measure
from skimage.measure import regionprops
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segment_score(img, y_true, y_pred, visualization=False, min_area=10):
    """

    :param img:
    :param y_true:
    :param y_pred:
    :param visualization:
    :param min_area:
    :return:

    Calculates segmentation score by counting centroids of predicted regions which are or are not into true regions
    Returns sensitivity (TP/P), errors (FP/P) and diffusion
    """


    h, w = img.shape
    im_true = y_true.reshape(h, w)
    im_pred = y_pred.reshape(h, w)

    labels_pred = measure.label(im_pred)
    regions_pred = regionprops(labels_pred)

    centroids = np.array([list(x.centroid) for x in regions_pred])
    centroids = centroids.astype(int)
    areas = np.array([x.area for x in regions_pred])
    centroids = centroids[areas > min_area]

    compar = im_true[(centroids[:, 0], centroids[:, 1])]
    centroids_T = centroids[compar == 1]
    centroids_F = centroids[compar == 0]

    TP = sum(compar)
    FP = sum(1-compar)

    labels_true = measure.label(im_true)
    regions_true = regionprops(labels_true)
    P = len(regions_true)

    sensitivity = round(float(TP)/float(P), 3)
    errors = round(float(FP)/float(P), 3)

    if visualization:
        fig1 = plt.figure()
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(im_pred, alpha=0.7)
        plt.hold(True)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.hold(True)
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')
        plt.title('Prediction, Sensitivity : %s , Errors : %s ' % (sensitivity, errors))


        fig2 = plt.figure()
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(im_true, alpha=0.7)
        plt.hold(True)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.hold(True)
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')
        plt.title('Ground Truth, Sensitivity : %s , Errors : %s ' % (sensitivity, errors))
        plt.show()

    return [sensitivity, errors]


def rejectOne_score(img, y_true, y_pred, visualization=False, min_area=10, show_diffusion = True):
    """
    Calculates segmentation score by keeping an only true centroids as TP.
    Excess of centroids is counted by diffusion (Excess/P)
    Returns sensitivity (TP/P), errors (FP/P) and diffusion
    """

    h, w = img.shape
    im_true = y_true.reshape(h, w)
    im_pred = y_pred.reshape(h, w)

    labels_pred = measure.label(im_pred)
    regions_pred = regionprops(labels_pred)

    centroids = np.array([list(x.centroid) for x in regions_pred])
    centroids = centroids.astype(int)
    areas = np.array([x.area for x in regions_pred])
    centroids = centroids[areas > min_area]

    labels_true = measure.label(im_true)
    regions_true = regionprops(labels_true)

    centroid_candidates = set([tuple(row) for row in centroids])

    centroids_T = []
    notDetected = []
    n_extra = 0

    for axon in regions_true:
        axon_coords = [tuple(row) for row in axon.coords]
        axon_center = (np.array(axon.centroid)).astype(int)
        centroid_match = set(axon_coords) & centroid_candidates
        centroid_candidates = centroid_candidates.difference(centroid_match)
        centroid_match = list(centroid_match)
        if len(centroid_match) != 0:
            diff = np.sum((centroid_match - axon_center)**2, axis=1)
            ind = np.argmin(diff)
            center = centroid_match[ind]
            centroids_T.append(center)
            n_extra += len(centroid_match)-1
        if len(centroid_match) == 0:
            notDetected.append(axon_center)

    centroids_F = list(centroid_candidates)
    P = len(regions_true)
    TP = len(centroids_T)
    sensitivity = round(float(TP)/P,3)

    FP = len(centroids_F)
    errors = round(float(FP)/P,3)

    centroids_F = np.array(centroids_F)
    centroids_T = np.array(centroids_T)
    notDetected = np.array(notDetected)
    diffusion = float(n_extra)/TP


    if visualization:

        plt.figure(1)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(im_pred, alpha=0.7)
        plt.hold(True)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.hold(True)
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')
        plt.hold(True)
        plt.scatter(notDetected[:, 1], notDetected[:, 0], color='y')
        plt.title('Prediction, Sensitivity : %s , Errors : %s ' % (sensitivity, errors))

        plt.figure(2)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(im_true, alpha=0.7)
        plt.hold(True)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.hold(True)
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')
        plt.title('Ground Truth, Sensitivity : %s , Errors : %s ' % (sensitivity, errors))
        plt.show()

    if show_diffusion:
        return [sensitivity, errors, round(diffusion,4)]
    else:
        return [sensitivity, errors]


def dice(img,y_true,y_pred):

    h, w = img.shape
    im_true = y_true.reshape(h, w)
    im_pred = y_pred.reshape(h, w)

    labels_true = measure.label(im_true)
    regions_true = regionprops(labels_true)

    labels_pred = measure.label(im_pred)
    regions_pred = regionprops(labels_pred)
    features = ['coords','area','dice']
    df = pd.DataFrame(columns=features)

    i=0
    for x_pred in regions_pred :
        centroid = (np.array(x_pred.centroid)).astype(int)
        if im_true[(centroid[0], centroid[1])] == 1:
            for x_true in regions_true:
               if centroid in x_true.coords:
                   A = np.zeros((img.shape[0], img.shape[1]))
                   B = np.zeros((img.shape[0], img.shape[1]))


                   A[x_pred.coords[:, 0], x_pred.coords[:, 1]] = 1
                   B[x_true.coords[:, 0], x_true.coords[:, 1]] = 1
                   intersect = float((sum(sum(B))))

                   D = intersect/(sum(sum(B))+ sum(sum(A)))
                   df.loc[i] = [x_pred.coords , x_pred.area, D]
                   break
        i+=1
    return df