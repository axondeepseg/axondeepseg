from skimage import measure
from skimage.measure import regionprops
import numpy as np
import pandas as pd


def score_analysis(img, groundtruth, prediction, visualization=False, min_area=2):
    """
    Calculates segmentation score by keeping an only true centroids as TP.
    Excess of centroids detected for a unique object is counted by diffusion (Excess/TP+FN)
    Returns sensitivity (TP/P), precision (TP/TP+FN) and diffusion

    :param img: image to segment
    :param groundtruth: groundtruth of the image
    :param prediction: segmentation
    :param visualization: if True, FP and TP are displayed on the image
    :param min_area: minimal area of the predicted axon to count.
    :return: [sensitivity, precision, diffusion]
    """

    labels_pred = measure.label(prediction)
    regions_pred = regionprops(labels_pred)

    centroids = np.array([list(x.centroid) for x in regions_pred])
    centroids = centroids.astype(int)
    areas = np.array([x.area for x in regions_pred])
    centroids = centroids[areas > min_area]

    labels_true = measure.label(groundtruth)
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
            diff = np.sum((centroid_match - axon_center) ** 2, axis=1)
            ind = np.argmin(diff)
            center = centroid_match[ind]
            centroids_T.append(center)
            n_extra += len(centroid_match) - 1
        if len(centroid_match) == 0:
            notDetected.append(axon_center)

    centroids_F = list(centroid_candidates)
    P = len(regions_true)
    TP = len(centroids_T)

    FP = len(centroids_F)

    centroids_F = np.array(centroids_F)
    centroids_T = np.array(centroids_T)
    # not_detected = len(np.array(notDetected))

    sensitivity = round(float(TP) / P, 3)
    # errors = round(float(FP) / P, 3)
    diffusion = float(n_extra) / (TP + FP)
    precision = round(float(TP) / (TP + FP), 3)

    if visualization:
        plt.figure(1)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(prediction, alpha=0.7)
        plt.hold(True)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.hold(True)
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')
        plt.hold(True)
        plt.scatter(notDetected[:, 1], notDetected[:, 0], color='y')
        plt.title('Prediction, Sensitivity : %s , Precision : %s ' % (sensitivity, precision))

        plt.figure(2)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(groundtruth, alpha=0.7)
        plt.hold(True)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.hold(True)
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')
        plt.title('Ground Truth, Sensitivity : %s , Precision : %s ' % (sensitivity, precision))
        plt.show()

    return [sensitivity, precision, round(diffusion, 4)]


def dice(img, groundtruth, prediction, min_area=3):
    """
    :param img: image to segment
    :param groundtruth : True segmentation
    :param prediction : Segmentation predicted by the algorithm
    :param min_area: minimum area of the predicted object to measure dice
    :return dice_scores: pandas dataframe associating the axon predicted, its size and its dice score

    To get the global dice score of the prediction,
    """

    h, w = img.shape

    labels_true = measure.label(groundtruth)
    regions_true = regionprops(labels_true)

    labels_pred = measure.label(prediction)
    regions_pred = regionprops(labels_pred)
    features = ['coords', 'area', 'dice']
    df = pd.DataFrame(columns=features)

    i = 0
    for axon_predicted in regions_pred:
        centroid = (np.array(axon_predicted.centroid)).astype(int)
        if groundtruth[(centroid[0], centroid[1])] == 1:
            for axon_true in regions_true:

                if [centroid[0], centroid[1]] in axon_true.coords.tolist():
                    surface_pred = np.zeros((h, w))
                    surface_true = np.zeros((h, w))

                    surface_pred[axon_predicted.coords[:, 0], axon_predicted.coords[:, 1]] = 1
                    surface_true[axon_true.coords[:, 0], axon_true.coords[:, 1]] = 1
                    intersect = surface_pred * surface_true

                    Dice = 2 * float(sum(sum(intersect))) / (sum(sum(surface_pred)) + sum(sum(surface_true)))
                    df.loc[i] = [axon_predicted.coords, axon_predicted.area, Dice]
                    break
        i += 1
    dice_scores = df[df['area'] > min_area]
    return dice_scores

def pw_dice(img1, img2):
    """
    img1 and img2 are boolean masks ndarrays
    This functions compute the pixel-wise dice coefficient (not axon-wise but pixel wise)
    """

    img_sum = img1.sum() + img2.sum()
    if img_sum == 0:
        return 1

    intersection = np.logical_and(img1, img2)
    # Return the global dice coefficient
    return 2. * intersection.sum() / img_sum