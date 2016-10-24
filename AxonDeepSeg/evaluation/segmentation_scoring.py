from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rejectOne_score(img, y_true, y_pred, visualization=False, min_area=2):
    """
    Calculates segmentation score by keeping an only true centroids as TP.
    Excess of centroids is counted by diffusion (Excess/TP+FN)
    Returns sensitivity (TP/P), precision (FP/TP+FN) and diffusion
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

    FP = len(centroids_F)

    centroids_F = np.array(centroids_F)
    centroids_T = np.array(centroids_T)
    not_detected = len(np.array(notDetected))

    sensitivity = round(float(TP) / P, 3)
    errors = round(float(FP) / P, 3)
    diffusion = float(n_extra)/(TP+FP)
    precision = round(float(TP)/(TP+FP), 3)

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

    return [sensitivity, precision, round(diffusion,4)]


def dice(img,y_true, y_pred, min_area = 9):
    """
    :param img: image to segment
    :param y_true: ground truth vectorized
    :param y_pred: prediction vectorized
    :param min_area: minimum area of the predicted object to measure dice
    :return: dataframe with the object, its size and its dice score
    """

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

               if [centroid[0], centroid[1]] in x_true.coords.tolist():

                   A = np.zeros((img.shape[0], img.shape[1]))
                   B = np.zeros((img.shape[0], img.shape[1]))

                   A[x_pred.coords[:, 0], x_pred.coords[:, 1]] = 1
                   B[x_true.coords[:, 0], x_true.coords[:, 1]] = 1
                   intersect = A*B

                   D = 2*float(sum(sum(intersect)))/(sum(sum(B))+sum(sum(A)))
                   df.loc[i] = [x_pred.coords, x_pred.area, D]
                   break
        i+=1
    return df[df['area']>min_area]