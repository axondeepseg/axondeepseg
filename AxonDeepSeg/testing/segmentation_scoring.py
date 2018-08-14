from skimage import measure
from skimage.measure import regionprops
import numpy as np
import pandas as pd
from skimage.morphology import binary_erosion, disk, label
from scipy.spatial.distance import directed_hausdorff
import sys
from sys import platform as _platform
if 'pytest' in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg') # Enforces mpl to not open new plot windows
elif _platform == "darwin": # Mac OSX
    import matplotlib as mpl
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import AxonDeepSeg.ads_utils

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
        plt.imshow(prediction, alpha=0.7)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')

        notDetected = np.array(notDetected) # Bug fix, was a list of an np array, which can't be sliced using integer index
        plt.scatter(notDetected[:, 1], notDetected[:, 0], color='y')
        plt.title('Prediction, Sensitivity : %s , Precision : %s ' % (sensitivity, precision))

        plt.figure(2)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.imshow(groundtruth, alpha=0.7)
        plt.scatter(centroids_T[:, 1], centroids_T[:, 0], color='g')
        plt.scatter(centroids_F[:, 1], centroids_F[:, 0], color='r')
        plt.title('Ground Truth, Sensitivity : %s , Precision : %s ' % (sensitivity, precision))
        plt.show()

    return [sensitivity, precision, round(diffusion, 4)]


def dice(img, groundtruth, prediction, min_area=3):
    """
    :param img: image to segment
    :param groundtruth : (bool) True segmentation
    :param prediction : (bool) Segmentation predicted by the algorithm
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


class Metrics_calculator: 
    
    def __init__(self, prediction_mask, groundtruth_mask):
        self.prediction_mask = prediction_mask
        self.groundtruth_mask = groundtruth_mask

    def pw_sensitivity(self):

        # Compute true positives count
        TP_count = np.logical_and(self.prediction_mask, self.groundtruth_mask).sum()

        # Compute positives count (TP+FN)
        P_count = self.groundtruth_mask.sum()
    
        # Compute sensitivity = TP/(TP+FN) = TP/P
        return np.true_divide(TP_count, P_count)
    
    def pw_precision(self):

        # Compute true positives count
        TP_count = np.logical_and(self.prediction_mask, self.groundtruth_mask).sum()

        # Compute false positives count
        FP_count = np.logical_and(self.prediction_mask == 1, self.groundtruth_mask == 0).sum()

        # Compute precision = TP/(TP+FP)
        return np.true_divide(TP_count, (TP_count + FP_count))       
    
    def pw_specificity(self):

        # Compute true negatives count
        TN_count = np.logical_and(self.prediction_mask == 0, self.groundtruth_mask == 0).sum()

        # Compute false positives count
        FP_count = np.logical_and(self.prediction_mask == 1, self.groundtruth_mask == 0).sum()

        # Compute specificity = TN/(TN+FP) = TN/N
        return np.true_divide(TN_count, (TN_count + FP_count))    
    
    def pw_FN_rate(self):

        # Compute false negative rate = 1 - sensitivity
        return (1 - self.pw_sensitivity())    
    
    def pw_FP_rate(self):

        # Compute false positive rate = 1 - specificity
        return (1 - self.pw_specificity())

    def pw_accuracy(self):

        # Compute true positives count
        TP_count = np.logical_and(self.prediction_mask, self.groundtruth_mask).sum()
    
        # Compute true negatives count
        TN_count = np.logical_and(self.prediction_mask == 0, self.groundtruth_mask == 0).sum()
    
        # Compute accuracy = (TP+TN)/total_count
        return np.true_divide((TN_count+TP_count), self.prediction_mask.size)

    def pw_F1_score(self):

        # Compute F1 score = 2*(precision*sensitivity)/(precision+sensitivity)
        return np.true_divide(2*(self.pw_sensitivity()*self.pw_precision()),(self.pw_sensitivity()+self.pw_precision()))

    def pw_dice(self):

        # Compute true positives count
        TP_count = np.logical_and(self.prediction_mask, self.groundtruth_mask).sum()
        
        # Compute false positives count
        FP_count = np.logical_and(self.prediction_mask == 1, self.groundtruth_mask == 0).sum()
    
        # Compute false negatives count
        FN_count = np.logical_and(self.prediction_mask == 0, self.groundtruth_mask == 1).sum()
    
        # Compute Dice = 2TP/(2TP+FP+FN)
        return np.true_divide(2*TP_count,2*TP_count+FP_count+FN_count)
    
    def pw_jaccard(self):

        # Compute true positives count
        TP_count = np.logical_and(self.prediction_mask, self.groundtruth_mask).sum()
        
        # Compute false positives count
        FP_count = np.logical_and(self.prediction_mask == 1, self.groundtruth_mask == 0).sum()
    
        # Compute false negatives count
        FN_count = np.logical_and(self.prediction_mask == 0, self.groundtruth_mask == 1).sum()
    
        # Compute Jaccard = TP/(TP+FP+FN)
        return np.true_divide(TP_count,TP_count+FP_count+FN_count)
    
    def ew_dice(self,output='short'):
        
        # Compute element-wise Dice for each object in masks
        ew_dice_raw = dice(self.prediction_mask, self.groundtruth_mask, self.prediction_mask, min_area=5)
        
        # Create dictionary with statistical metrics of the Dice distribution
        dice_list = {'mean': np.mean(ew_dice_raw.dice), 'std': np.std(ew_dice_raw.dice),
                     'min': np.min(ew_dice_raw.dice), 'max': np.max(ew_dice_raw.dice),
                     'percent_5': np.percentile(ew_dice_raw.dice,5), 'percent_10': np.percentile(ew_dice_raw.dice,10),
                     'percent_25': np.percentile(ew_dice_raw.dice,25), 'percent_50': np.percentile(ew_dice_raw.dice,50),
                     'percent_75': np.percentile(ew_dice_raw.dice,75), 'percent_90': np.percentile(ew_dice_raw.dice,90),
                     'percent_95': np.percentile(ew_dice_raw.dice,95)}
        
        if output=='all':
            return ew_dice_raw.dice
        if output=='short':
            return dice_list

    def pw_hausdorff_distance(self):
        
        # Compute small erosion on both masks
        str_elem = disk(3)
        pred_eroded = binary_erosion(self.prediction_mask, str_elem)
        gt_eroded = binary_erosion(self.groundtruth_mask, str_elem)
        
        # Get outer contours
        pred_contour = np.logical_and(pred_eroded == 0 , self.prediction_mask == 1)
        gt_contour = np.logical_and(gt_eroded == 0 , self.groundtruth_mask == 1)
        
        # Compute global hausdorff distance metric
        return directed_hausdorff(pred_contour, gt_contour)[0]
