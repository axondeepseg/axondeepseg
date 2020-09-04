# This files is used to generate statistics on a select sample of images, using the specified model.

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import pickle
from AxonDeepSeg.apply_model import axon_segmentation
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, log_loss
from AxonDeepSeg.testing.segmentation_scoring import pw_dice
import time
from AxonDeepSeg.config_tools import rec_update
import pandas as pd
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.ads_utils import convert_path

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

    # If string, convert to Path objects
    path_model_folder = convert_path(path_model_folder)
    path_images_folder = convert_path(path_images_folder)

    # First we load every information independent of the model
    # We generate the list of testing folders, each one containing one image
    images_folders = filter(lambda p: p.is_dir(), path_images_folder.iterdir())
    path_images_folder = [path_images_folder / x for x in images_folders]

    # We check that the model path we were given exists.
    if path_model_folder.is_dir():

        # Generation of statistics
        stats_dict = generate_statistics(path_model_folder, path_images_folder, resampled_resolution, overlap_value, verbosity_level=verbosity_level)

        # We now save the data in a corresponding json file.
        save_metrics(stats_dict, path_model_folder, statistics_filename)

        # Finally we print out the results using pprint.
        print_metrics(stats_dict)


def metrics_single_wrapper(path_model_folder, path_images_folder, resampled_resolution, overlap_value=25,
                            statistics_filename='model_statistics_validation.json',
                            create_statistics_file=True, verbosity_level=0):
    """
    Procedure to compute the metrics using a model on several images. Computation is made on one image after the other.
    :param path_model_folder: Path to the folder where the model is located.
    :param path_images_folder: Path to the folders that contain the images to compute the metrics on.
    :param resampled_resolution: Float, the resolution to resample to to make the predictions.
    :param overlap_value: Int, the number of pixels to use for overlap.
    :param statistics_filename: String, the name of the file to use when saving the computed metrics.
    :param create_statistics_file: Boolean. If true, creates a statistics file where the computed metrics are stored.
    :param verbosity_level: Int. The higher, the more displayed information.
    :return: Nothing.
    """

    # If string, convert to Path objects
    path_model_folder = convert_path(path_model_folder)
    path_images_folder = convert_path(path_images_folder)

    # First we load every information independent of the model
    # We generate the list of testing folders, each one containing one image
    images_folders = filter(lambda p: p.is_dir(), path_images_folder.iterdir())
    path_images_folder = [path_images_folder / x for x in images_folders]

    # We check that the model path we were given exists.
    if path_model_folder.is_dir():

        # Generation of statistics for each image one after the other
        for current_path_images_folder in path_images_folder:

            stats_dict = generate_statistics(path_model_folder, [current_path_images_folder],
                                             resampled_resolution, overlap_value, verbosity_level=verbosity_level)

            # We now save the data in a corresponding json file.
            save_metrics(stats_dict, path_model_folder, statistics_filename)

            # Finally we print out the results using pprint.
            print_metrics(stats_dict)





def print_metrics(metrics_dict, filter_ckpt=None):
    """
    Displays the computed metrics entered as input in the terminal.
    :param metrics_dict: Dictionary, contains the computed metrics.
    :param filter_ckpt: String, name of the checkpoint to use. If None, using all checkpoints.
    :return: Nothing.
    """

    dict_ckpt = metrics_dict["data"]

    if filter_ckpt != None:

        # We go through every checkpoint in the list and we only return the checkpoint dictionary which name is the
        # one we want to filter.
        for ckpt_elem in list(dict_ckpt.values()):
            if ckpt_elem['ckpt'] == str(filter_ckpt):
                dict_ckpt = [ckpt_elem]
                break

    for current_ckpt in list(dict_ckpt.values()):

        print(("Model: " + str(current_ckpt["id_model"]) + \
              ", ckpt: " + str(current_ckpt["ckpt"]) + ", date: " + str(metrics_dict["date"])))

        for name_image, test_image_stats in list(current_ckpt["testing_stats"].items()):

            t = PrettyTable(["Metric", "Value"])
            t.add_row(["name_image", name_image])

            for key, value in list(test_image_stats.items()):
                t.add_row([key, value])
            print(t)



def generate_statistics(path_model_folder, path_images_folder, resampled_resolution, overlap_value,
                        verbosity_level=0):
    """
    Generates the implemented statistics for all the checkpoints of a given model, for each requested image.
    :param path_model_folder: Path to the model to use.
    :param path_images_folder: Path to the folders that contain the images to compute the metrics on.
    :param resampled_resolution: Float, the resolution to resample to to make the predictions.
    :param overlap_value: Int, the number of pixels to use for overlap.
    :param verbosity_level: Int. The higher, the more displayed information.
    :return:
    """
    print(path_images_folder)
    # If string, convert to Path objects
    path_model_folder = convert_path(path_model_folder)
    path_images_folder = convert_path(path_images_folder)
    print(path_images_folder)

    model_statistics_dict = {"date":time.strftime("%Y-%m-%d"),
                             "data":{}}
    # First we load the network parameters from the config file
    with open(path_model_folder / 'config_network.json', 'r') as fd:
        config_network = json.loads(fd.read())

    n_classes = config_network['n_classes']
    model_name = path_model_folder.parts[-2] # Extraction of the name of the model.

    # We loop over all checkpoint files to compute statistics for each checkpoint.
    for checkpoint in path_model_folder.iterdir():

        if str(checkpoint)[-10:] == '.ckpt.meta':

            result_model = {}
            name_checkpoint = str(checkpoint)[:-10]

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
                print('No stats file found...')
                #f = open(path_model_folder + '/evolution.pkl', 'r')
                #res = pickle.load(f)
                #epoch_stats = max(res['steps'])
                #acc_stats = np.mean(res['accuracy'][-10:])
                #loss_stats = np.mean(res['loss'][-10:])

                epoch_stats = None
                acc_stats = None
                loss_stats = None

            result_model.update({
                'training_stats': {
                'training_epoch': epoch_stats,
                'training_mvg_avg10_acc': acc_stats,
                'training_mvg_avg10_loss': loss_stats
            },
                'testing_stats': {}
            })

            # 2/ Computation of the predictions / outputs of the network for each image at the same time.

            predictions, outputs_network = axon_segmentation(path_images_folder,
                                                            ['image.png']*len(path_images_folder),
                                                            path_model_folder,
                                                            config_network, ckpt_name=name_checkpoint,
                                                            overlap_value=overlap_value,
                                                            resampled_resolutions=[resampled_resolution]*len(path_images_folder),
                                                            prediction_proba_activate=True,
                                                            write_mode=False,
                                                            gpu_per=1.0,
                                                            verbosity_level=verbosity_level
                                                            )
            # These two variables are list, as long as the number of images that are tested.

            if verbosity_level>=2:
                print('Statistics extraction...')

            # 3/ Computation of the statistics for each image.
            for i, image_folder in tqdm(enumerate(path_images_folder)):

                current_prediction = predictions[i]
                current_network_output = outputs_network[i]

                # Reading the images and processing them
                mask_raw = ads.imread(image_folder / 'mask.png')
                mask = labellize(mask_raw)

                # We infer the name of the different files
                name_image = image_folder.name

                # Computing metrics and storing them in the json file.
                current_proba = output_network_to_proba(current_network_output, n_classes)
                testing_stats_dict = compute_metrics(current_prediction, current_proba, mask, n_classes)
                result_model['testing_stats'].update({name_image:testing_stats_dict})

            # We add the metrics for all the checkpoints from this model (on all images) to the data list.
            model_statistics_dict["data"].update({name_checkpoint:result_model})

    return model_statistics_dict


def save_metrics(model_statistics_dict, path_model_folder, statistics_filename):
    """
    Saves the computed metrics in a json file.
    :param model_statistics_dict: Dict, the computed metrics to save.
    :param path_model_folder: Path to the folder containing the model to use.
    :param statistics_filename: Name of the file where we will store the computed statistics.
    :return:
    """

    # If string, convert to Path objects
    path_model_folder = convert_path(path_model_folder)

    # If the file already exists we rename the old one with a .old suffix.
    path_statistics_file = path_model_folder / statistics_filename

    if path_statistics_file.exists():
        with open(path_statistics_file) as f:
            original_stats_dict = json.load(f)
            path_statistics_file.unlink()
    else:
        original_stats_dict = {}

    #original_stats_dict.update(model_statistics_dict)
    rec_update(original_stats_dict, model_statistics_dict)


    with open(path_statistics_file, 'w') as f:
        json.dump(original_stats_dict, f, indent=2)


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

def binarize(mask_raw):
    vals = np.unique(mask_raw)
    mask = np.zeros((mask_raw.shape[0], mask_raw.shape[1], len(vals)))
    for i, e in enumerate(vals):
        mask[:, :, i] = mask_raw == e
    return mask

def volumize(mask_labellized, n_class):
    '''
    :param mask_labellized: 2-D array with each class being indicated as its corresponding
    number. ex : [[0,0,1],[2,2,0],[0,1,2]].
    '''
    mask = np.zeros((mask_labellized.shape[0], mask_labellized.shape[1], n_class))

    for i in range(n_class):
        mask[:, :, i] = mask_labellized == i

    return mask

# Aggregation of statistics.


class metrics():
    """
    We use this class to manage metrics and create easily aggregation. We then are able to save them in csv format.
    """
    def __init__(self, statistics_filename='model_statistics_validation.json'):
        self.statistics_filename = statistics_filename
        self.path_models = set()
        self.stats = pd.DataFrame()
        self.filtered_stats = pd.DataFrame()
        self.aggregated_stats = pd.DataFrame()
        self.columns = ['id_model', 'ckpt', 'type_model', 'type_image', 'pw_dice_myelin',
                        'pw_dice_axon', 'testing_log_loss', 'testing_accuracy', 'testing_name_image']

    def add_models(self, path_models):
        if type(path_models) != list:
            path_models = [path_models]
        [self.path_models.add(e) for e in path_models]

    def load_models(self):
        for path in self.path_models:
            
            # If string, convert to Path objects
            path = convert_path(path)

            try:
                with open(path / self.statistics_filename) as f:
                    stats_dict = json.loads(f.read())['data']
            except:
                raise ValueError('No config file found: statistics json file missing in the model folder.')

            # Now we add a line to the stats dataframe for each model
            for ckpt_name, ckpt in list(stats_dict.items()):

                # Getting each part of data
                model_name = ckpt['id_model']
                ckpt_name = ckpt['ckpt']
                config = ckpt['config']
                testing_stats_list = ckpt['testing_stats']
                for name_image, testing_stats in list(testing_stats_list.items()):
                    type_image = name_image.split("_")[0]
                    pw_dice_myelin = testing_stats['pw_dice_myelin']
                    pw_dice_axon = testing_stats['pw_dice_axon']
                    testing_log_loss = testing_stats['log_loss']
                    testing_accuracy = testing_stats['accuracy']

                    new_line = [[model_name, ckpt_name, config['trainingset'].split('_')[0],
                                 type_image, pw_dice_myelin, pw_dice_axon,
                                 testing_log_loss, testing_accuracy, name_image]]

                    # Updating the dataframe with the latest data
                    self.stats = self.stats.append(pd.DataFrame(columns=self.columns, data=new_line))

                self.filtered_stats = self.stats.copy()

    def filter_(self, list_acquisitions=None, list_ckpt=None, write_mode=False, name_file=None):
        filtered_stats = pd.DataFrame()

        if list_acquisitions != None:
            # Processing arguments
            if type(list_acquisitions) != list:
                list_acquisitions = [list_acquisitions]

            # For each acquisition type
            for image_to_take in list_acquisitions:
                filtered_stats = filtered_stats.append(self.stats.loc[self.stats['type_image'] == image_to_take])
        if list_ckpt != None:
            # Processing arguments
            if type(list_ckpt) != list:
                list_ckpt = [list_ckpt]
            for ckpt in list_ckpt:
                filtered_stats = filtered_stats.append(self.stats.loc[self.stats['ckpt'] == ckpt])
        self.filtered_stats = filtered_stats

        if write_mode == True:
            if name_file is None:
                name_file = 'filtered_' + '_'.join(list_acquisitions) + '_' + time.strftime("%Y-%m-%d") + '.csv'
            filtered_stats.T.to_csv(name_file)

        # Outputting the filtered pandas dataframe.
        return filtered_stats

    def aggregate(self, list_metrics, write_mode=False, name_file=None):
        # Processing arguments
        aggregated_stats = pd.DataFrame()
        if type(list_metrics) != list:
            list_metrics = [list_metrics]

        for metric in list_metrics:
            tmp = self.filtered_stats.groupby(['id_model', 'ckpt']).apply(metric)
            tmp.columns = [x + '_' + metric.__name__ for x in tmp.columns.tolist()]
            aggregated_stats = pd.concat([aggregated_stats, tmp],
                                         axis=1, ignore_index=False)

        if write_mode == True:
            if name_file is None:
                name_file = 'agg_' + '_'.join([x.__name__ for x in list_metrics]) + '_' + time.strftime(
                    "%Y-%m-%d") + '.csv'
            aggregated_stats.T.to_csv(name_file)

        return aggregated_stats

# ARGUMENTS

# m path model
# p path data
# t type of acquisition
# s resampling resolution
# o overlap value

def main():

    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--model", required=False, default="../models/defaults/default_SEM_model/")
    ap.add_argument("-d", "--data", required=False, default="../../data/baseline_validation/")
    ap.add_argument("-t", "--type", required=False, default="single")
    ap.add_argument("-r", "--resolution", required=False, default=0.1)

    args = vars(ap.parse_args())

    path_model = str(args["model"])
    path_data = str(args["data"])
    type_computation = str(args["type"])
    resampling_resolution = float(args["resolution"])


    if type_computation == "single":
        metrics_single_wrapper(path_model, path_data, resampling_resolution)

    else:
        metrics_classic_wrapper(path_model, path_data, resampling_resolution)

if  __name__ == '__main__':
    main()
