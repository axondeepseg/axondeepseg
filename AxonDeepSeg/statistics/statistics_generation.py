# This files is used to generate statistics on a select sample of images, using the specified model.

import os, json
import numpy as np
from tqdm import tqdm
import pickle
from AxonDeepSeg.apply_model import axon_segmentation
from scipy.misc import imread
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, log_loss
from AxonDeepSeg.testing.segmentation_scoring import pw_dice
import shutil
import time

def metrics_classic_wrapper(path_model_folder, path_images_folder, resampled_resolution, overlap_value=25,
                            statistics_filename='model_statistics_validation.json',
                            create_statistics_file=True, verbosity_level=0):

    """
    Procedure to compute metrics on all the images we want, at the same time.
    :param path_model_folder: Path to the (only) model we want to use for statistics generation .
    :param path_images_folder: Path to the folder containing all the folders containing the images (each image has its
    own folder).
    :param resampled_resolution: The size in micrometer of a pixel we resample to.
    :param overlap_value: The number of pixels used for overlap.
    :param statistics_filename: String, the file name to use when creating a statistics file.
    :param create_statistics_file: Boolean. If False, the function just displays the statistics. True by default.
    :param verbosity_level: Int. The higher, the more information displayed about the metrics computing process.
    :return: Nothing.
    """

    # First we load every information independent of the model
    # We generate the list of testing folders, each one containing one image
    images_folders = [d for d in os.listdir(path_images_folder) if os.path.isdir(os.path.join(path_images_folder, d))]
    path_images_folder = map(lambda x: os.path.join(path_images_folder, x), images_folders)

    # We check that the model path we were given exists.
    if os.path.isdir(path_model_folder):

        # We check that there is no saved statistics json file with the same name.
        if not os.path.exists(os.path.join(path_model_folder, statistics_filename)):

            # Generation of statistics
            stats_dict = generate_statistics(path_model_folder, path_images_folder, resampled_resolution, overlap_value)

            # We now save the data in a corresponding json file.
            save_metrics(stats_dict, path_model_folder, statistics_filename)

            # Finally we print out the results using pprint.
            print_metrics(stats_dict)


# TODO : function like the one above that computes the statistics for one image after the other.

def print_metrics(metrics_dict, filter_ckpt=None):


    list_ckpt = metrics_dict["data"]

    if filter_ckpt != None:

        # We go through every checkpoint in the list and we only return the checkpoint dictionary which name is the
        # one we want to filter.
        for ckpt_elem in list_ckpt:
            if ckpt_elem['ckpt'] == str(filter_ckpt):
                list_ckpt = [ckpt_elem]
                break

    for current_ckpt in list_ckpt:
        print "Model: " + str(current_ckpt["id_model"]) + \
              ", ckpt: " + str(current_ckpt["ckpt"]) + ", date: " + str(metrics_dict["date"])
        for test_image_stats in current_ckpt["testing_stats"]:

            t = PrettyTable(["Metric", "Value"])

            for key, value in test_image_stats.iteritems():
                t.add_row([key, value])
            print t



def generate_statistics(path_model_folder, path_images_folder, resampled_resolution, overlap_value,
                        verbosity_level=0):
    """
    Generates the implemented statistics for all the checkpoints of a given model, for each requested image.
    :param path_model_folder:
    :param path_images_folder:
    :param resampled_resolution:
    :param overlap_value:
    :param verbosity_level:
    :return:
    """

    model_statistics_dict = {"date":time.strftime("%Y-%m-%d"),
                             "data":[]}
    # First we load the network parameters from the config file
    with open(os.path.join(path_model_folder, 'config_network.json'), 'r') as fd:
        config_network = json.loads(fd.read())

    trainingset_name = config_network['trainingset']
    type_trainingset = trainingset_name.split('_')[0]
    n_classes = config_network['n_classes']
    model_name = path_model_folder.split("/")[-1] # Extraction of the name of the model.

    # We loop over all checkpoint files to compute statistics for each checkpoint.
    for checkpoint in os.listdir(path_model_folder):

        if checkpoint[-10:] == '.ckpt.meta':

            result_model = {}
            name_checkpoint = checkpoint[:-10]

            result_model.update({'id_model': model_name,
                                 'ckpt': name_checkpoint,
                                 'config': config_network})

            # 1/ We load the saved training statistics, which are independent of the testing images

            try:
                f = open(path_model_folder + '/' + name_checkpoint + '.pkl', 'r')
                res = pickle.load(f)
                acc_stats = res['accuracy']
                loss_stats = res['loss']
                epoch_stats = res['steps']

            except:
                print 'No stats file found...'
                f = open(path_model_folder + '/evolution.pkl', 'r')
                res = pickle.load(f)
                epoch_stats = max(res['steps'])
                acc_stats = np.mean(res['accuracy'][-10:])
                loss_stats = np.mean(res['loss'][-10:])

            result_model.update({
                'training_stats': {
                'training_epoch': epoch_stats,
                'training_mvg_avg10_acc': acc_stats,
                'training_mvg_avg10_loss': loss_stats
            },
                'testing_stats': []
            })

            # 2/ Computation of the predictions / outputs of the network for each image at the same time.s

            predictions, outputs_network = axon_segmentation([path_images_folder], ['image.png'], path_model_folder,
                                                            config_network, ckpt_name=name_checkpoint,
                                                            overlap_value=overlap_value,
                                                            resampled_resolutions=[resampled_resolution],
                                                            prediction_proba_activate=True,
                                                            write_mode=False,
                                                            gpu_per=1.0,
                                                            verbosity_level=verbosity_level
                                                            )
            # These two variables are list, as long as the number of images that are tested.

            if verbosity_level>=2:
                print 'Statistics extraction...'

            # 3/ Computation of the statistics for each image.
            for i, image_folder in tqdm(enumerate(path_images_folder)):

                path_image_folder = os.path.join(path_images_folder, image_folder)
                current_prediction = predictions[i]
                current_network_output = outputs_network[i]

                # Reading the images and processing them
                mask_raw = imread(os.path.join(path_image_folder, 'mask.png'), flatten=True, mode='L')
                mask = labellize(mask_raw)

                # We infer the name of the different files
                type_image = image_folder.split('_')[0]  # SEM or TEM
                name_image = '_'.join(image_folder.split('_')[1:])  # Rest of the name of the image
                testing_stats_dict = {'type_image': type_image,
                                      'name_image': name_image}

                # Computing metrics and storing them in the json file.
                current_proba = output_network_to_proba(current_network_output)
                testing_stats_dict.update(compute_metrics(current_prediction, current_proba, mask, n_classes))
                result_model['testing_stats'].append(testing_stats_dict)

            # We add the metrics for all the checkpoints from this model (on all images) to the data list.
            model_statistics_dict["data"].append(result_model)

    return model_statistics_dict


def save_metrics(model_statistics_dict, path_model_folder, statistics_filename):


    # TODO : update the function so that it also works when we do one image at a time.

    # If the file already exists we rename the old one with a .old suffix.
    path_statistics_file = os.path.join(path_model_folder, statistics_filename)

    if os.path.exists(path_statistics_file):
        shutil.move(path_statistics_file, os.path.join(path_model_folder, statistics_filename+'.old'))

    with open(path_statistics_file, 'w') as f:
        json.dump(model_statistics_dict, f, indent=2)


def output_network_to_proba(output_network, n_classes):
    """
    Softmax function applied to the output of the network.
    :param output_network: The pre-activation outputted by the network (function uconv_net), before applying softmax.
    :return: Tensor, same shape as output_network, but probabilities instead.
    """
    a = np.exp(output_network)
    b = np.sum(a, axis=-1)
    return np.stack([np.divide(a[:, :, l], b) for l in range(n_classes)], axis=-1)


def compute_metrics(prediction, proba, mask, n_classes):
    # Generate all the statistics for the current image using the current model, and stores them in a dictionary.

    stats = {}

    # Computation of intermediary metrics which will be used for computing final metrics.
    vec_prediction = np.reshape(volumize(prediction, n_classes), (-1, n_classes))
    vec_pred_proba = np.reshape(proba, (-1, n_classes))
    vec_mask = np.reshape(volumize(mask, n_classes), (-1, n_classes))
    gold_standard_axons = volumize(mask, n_classes)[:, :, -1]
    prediction_axon = volumize(prediction, n_classes)[:, :, -1]

    # >> Accuracy and XEntropy loss
    stats.update({
        'accuracy': accuracy_score(mask.ravel(), prediction.ravel()),
        'log_loss': log_loss(vec_mask, vec_pred_proba)
    })
    # >> Pixel wise dice, both classes
    pw_dice_axon = pw_dice(prediction_axon, gold_standard_axons)
    stats.update({
        'pw_dice_axon': pw_dice_axon})

    if n_classes == 3:
        gt_myelin = volumize(mask, n_classes)[:, :, 1]
        pred_myelin = volumize(prediction, n_classes)[:, :, 1]
        pw_dice_myelin = pw_dice(pred_myelin, gt_myelin)

        stats.update({
            'pw_dice_myelin': pw_dice_myelin})

    return stats


def labellize(mask_raw, thresh=[0, 0.2, 0.8]): # TODO : check function
    max_ = np.max(mask_raw)
    n_c = len(thresh)

    mask = np.zeros_like(mask_raw)
    for i, e in enumerate(thresh[1:]):
        mask[np.where(mask_raw >= e * 255)] = i + 1

    return mask

def binarize(mask_raw): # TODO : check function
    vals = np.unique(mask_raw)
    mask = np.zeros((mask_raw.shape[0], mask_raw.shape[1], len(vals)))
    for i, e in enumerate(vals):
        mask[:, :, i] = mask_raw == e
    return mask

def volumize(mask_labellized, n_class):# TODO : check function
    '''
    :param mask_labellized: 2-D array with each class being indicated as its corresponding
    number. ex : [[0,0,1],[2,2,0],[0,1,2]].
    '''
    mask = np.zeros((mask_labellized.shape[0], mask_labellized.shape[1], n_class))

    for i in range(n_class):
        mask[:, :, i] = mask_labellized == i

    return mask

# ARGUMENTS

# m path model
# p path data
# t type of acquisition
# s resampling resolution
# o overlap value

def main():

    metrics_classic_wrapper("../../models/defaults/default_SEM_model_v1/",
                            "../../data/baseline_validation/",
                            0.1)

if  __name__ == '__main__':
    main()