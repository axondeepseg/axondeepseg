# -*- coding: utf-8 -*-
from skimage.transform import rescale, resize
import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.ads_utils import convert_path
from AxonDeepSeg.network_construction import *
from AxonDeepSeg.visualization.get_masks import get_masks
from AxonDeepSeg.patch_management_tools import im2patches_overlap, patches2im_overlap
from AxonDeepSeg.config_tools import update_config, default_configuration

#Keras import
from keras import backend as K


def apply_convnet(path_acquisitions, acquisitions_resolutions, path_model_folder, config_dict, ckpt_name='model',
                  inference_batch_size=1, overlap_value=25, resampled_resolutions=[0.1],
                  prediction_proba_activate=False, gpu_per=1.0, verbosity_level=0):
    """
    Preprocesses the images, transform them into patches, applies the network, stitches the predictions and return them.
    :param path_acquisitions: List of path to the acquisitions.
    :param acquisitions_resolutions: List of the acquisitions resolutions (floats).
    :param path_model_folder: Path to the model folder.
    :param config_dict: Dictionary containing the model's parameters.
    :param ckpt_name: String, checkpoint to use.
    :param inference_batch_size: Int, batch size to use when doing inference.
    :param overlap_value: Int, number of pixels to use when overlapping the predictions of the network.
    :param resampled_resolutions: List of resolutions (flaots) to resample to before performing inference.
    :param prediction_proba_activate: Boolean, whether to compute the probability maps or not.
    :param gpu_per: Float, percentage of GPU to use if we use it.
    :param verbosity_level: Int, how much information to display.
    :return: List of segmentations, and list of probability maps if requested.
    """

    # If string, convert to Path objects
    path_acquisitions = convert_path(path_acquisitions)
    path_model_folder = convert_path(path_model_folder)

    # We set the logging from python and Tensorflow to a high level, to avoid messages
    # in the console when performing segmentation.
    from logging import ERROR
    tf.logging.set_verbosity(ERROR)
    import warnings
    warnings.filterwarnings('ignore')

    # Network Parameters
    patch_size = config_dict["trainingset_patchsize"]
    n_classes = config_dict["n_classes"]

    # STEP 1: Load and rescale the acquisitions, and transform them into patches.

    rs_acquisitions, rs_coeffs, original_acquisitions_shapes = load_acquisitions(
        path_acquisitions, acquisitions_resolutions, resampled_resolutions, verbose_mode=verbosity_level)

    # If we are unable to load the model, we return an error message
    if not path_model_folder.exists():
        print('Error: unable to find the requested model.')
        return [None] * len(path_acquisitions)

    L_data, L_n_patches, L_positions = prepare_patches(rs_acquisitions, patch_size, overlap_value)

    # STEP 2: Construct Tensorflow's computing graph and restoration of the session

    # Construction of the graph
    if verbosity_level >= 2:
        print("Graph construction ...")

    x = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 1))

    model = uconv_net(config_dict, bn_updated_decay=None, verbose=True)  # inference
    pred = model.output


    saver = tf.train.Saver()  # Load previous model

    # We limit the amount of GPU for inference
    config_gpu = tf.ConfigProto(log_device_placement=False)
    config_gpu.gpu_options.per_process_gpu_memory_fraction = gpu_per

    # Launch the session (this part takes time). All images will be processed by loading the session just once.

    sess = tf.Session(config=config_gpu)
    K.set_session(sess)

    model_previous_path = path_model_folder.joinpath(ckpt_name).with_suffix('.ckpt')
    saver.restore(sess, str(model_previous_path))

    # STEP 3: Inference

    if verbosity_level >= 2:
        print("Beginning inference ...")

    n_patches = len(L_data)
    it, rem = divmod(n_patches, inference_batch_size)

    predictions_list = []
    predictions_proba_list = []

    # Inference of complete batches
    for i in range(it):

        if verbosity_level >= 3:
            print(('processing patch %s of %s' % (i + 1, it)))

        batch_x = np.array(L_data[i * inference_batch_size:(i + 1) * inference_batch_size], dtype=np.uint8)

        if prediction_proba_activate:

            # First we perform inference on the input.
            current_batch_prediction, current_batch_prediction_proba = perform_batch_inference(
                model, sess, pred, x, batch_x, inference_batch_size, patch_size,
                n_classes, prediction_proba_activate=prediction_proba_activate)

            # Update of the predictions lists.
            predictions_list.extend(current_batch_prediction)
            predictions_proba_list.extend(current_batch_prediction_proba)

        else:
            current_batch_prediction = perform_batch_inference(model, sess, pred, x, batch_x, inference_batch_size,
                                                               patch_size, n_classes,
                                                               prediction_proba_activate=prediction_proba_activate)
            # Update of the predictions lists.
            predictions_list.extend(current_batch_prediction)

    # Last batch if needed

    if rem != 0:

        if verbosity_level >= 4:
            print('processing last patch')

        batch_x = np.asarray(L_data[it * inference_batch_size:])

        if prediction_proba_activate:

            # First we perform inference on the input.
            current_batch_prediction, current_batch_prediction_proba = perform_batch_inference(model,
                                                                                               sess, pred, batch_x, rem,
                                                                                               patch_size,
                                                                                               n_classes,
                                                                                               prediction_proba_activate=prediction_proba_activate)

            # Update of the predictions lists.
            predictions_list.extend(current_batch_prediction)
            predictions_proba_list.extend(current_batch_prediction_proba)

        else:
            current_batch_prediction = perform_batch_inference(model, sess, pred, batch_x, rem,
                                                               patch_size, n_classes,
                                                               prediction_proba_activate=prediction_proba_activate)
            # Update of the predictions lists.
            predictions_list.extend(current_batch_prediction)

    # End of the inference step.
    tf.reset_default_graph()

    # Now we have to transform the list of predictions in list of lists,
    # one for each full image : we put in each sublist the patches corresponding to a full image.

    ########### STEP 4: Reconstruction of the segmented patches into segmentations of acquisitions and
    # resampling to the original size

    if prediction_proba_activate:

        predictions, predictions_proba = process_segmented_patches(predictions_list, L_n_patches, L_positions,
                                                                   original_acquisitions_shapes,
                                                                   overlap_value, n_classes,
                                                                   predictions_proba_list=predictions_proba_list,
                                                                   prediction_proba_activate=prediction_proba_activate,
                                                                   verbose_mode=0)

        return predictions, predictions_proba

    else:
        predictions = process_segmented_patches(predictions_list, L_n_patches, L_positions,
                                                original_acquisitions_shapes,
                                                overlap_value, n_classes,
                                                predictions_proba_list=None,
                                                prediction_proba_activate=prediction_proba_activate,
                                                verbose_mode=0)

        return predictions

        #######################################################################################################################


def axon_segmentation(path_acquisitions_folders, acquisitions_filenames, path_model_folder, config_dict,
                      ckpt_name='model',
                      segmentations_filenames=['AxonDeepSeg.png'], inference_batch_size=1,
                      overlap_value=25, resampled_resolutions=0.1, acquired_resolution=None,
                      prediction_proba_activate=False, write_mode=True, gpu_per=1.0, verbosity_level=0):
    """
    Wrapper performing the segmentation of all the requested acquisitions and generates (if requested) the segmentation
    images.
    :param path_acquisitions_folders: List of folders where the acquisitions to segment are located.
    :param acquisitions_filenames: List of names of acquisitions to segment.
    :param path_model_folder: Path to the folder where the model is located.
    :param config_dict: Dictionary containing the configuration of the training parameters of the model.
    :param ckpt_name: String, name of the checkpoint to use.
    :param segmentations_filenames: List of the names of the segmentations files, to be used when creating the files.
    :param inference_batch_size: Size of the batches fed to the network.
    :param overlap_value: Int, number of pixels to use for overlapping the predictions.
    :param resampled_resolutions: List of the resolutions (in µm) to resample to.
    :param acquired_resolution: List of the resolutions (in µm) for native images.
    :param prediction_proba_activate: Boolean, whether to compute probability maps or not.
    :param write_mode: Boolean, whether to create segmentation images or not.
    :param gpu_per: Percentage of the GPU to use, if we use it.
    :param verbosity_level: Int, level of verbosity. The higher, the more information is displayed.
    :return: List of predictions, and optionally of probability maps.
    """

    # If string, convert to Path objects
    path_acquisitions_folders = convert_path(path_acquisitions_folders)
    path_model_folder = convert_path(path_model_folder)

    # Processing input so they are lists in every situation
    path_acquisitions_folders, acquisitions_filenames, resampled_resolutions, segmentations_filenames = \
        list(map(ensure_list_type, [path_acquisitions_folders, acquisitions_filenames, resampled_resolutions,
                                    segmentations_filenames]))

    if len(segmentations_filenames) != len(path_acquisitions_folders):
        segmentations_filenames = ['AxonDeepSeg.png'] * len(path_acquisitions_folders)

    if len(acquisitions_filenames) != len(path_acquisitions_folders):
        acquisitions_filenames = ['image.png'] * len(path_acquisitions_folders)

    if len(resampled_resolutions) != len(path_acquisitions_folders):
        resampled_resolutions = [resampled_resolutions[0]] * len(path_acquisitions_folders)

    # Generating the patch to acquisitions and loading the acquisitions resolutions.
    path_acquisitions = [path_acquisitions_folders[i] / e for i, e in enumerate(acquisitions_filenames)]

    # If we did not receive any resolution we read the pixel size in micrometer from each pixel.
    if acquired_resolution == None:
        if (path_acquisitions_folders[0] / 'pixel_size_in_micrometer.txt').exists():
            resolutions_files = [open(path_acquisition_folder / 'pixel_size_in_micrometer.txt', 'r')
                                 for path_acquisition_folder in path_acquisitions_folders]
            acquisitions_resolutions = [float(file_.read()) for file_ in resolutions_files]
        else:
            exception_msg = "ERROR: No pixel size is provided, and there is no pixel_size_in_micrometer.txt file in image folder. " \
                            "Please provide a pixel size (using argument -s), or add a pixel_size_in_micrometer.txt file " \
                            "containing the pixel size value."
            raise Exception(exception_msg)

    # If resolution is specified as input argument, use it
    else:
        acquisitions_resolutions = [acquired_resolution] * len(path_acquisitions_folders)

    # Ensuring that the config file is valid
    config_dict = update_config(default_configuration(), config_dict)

    # Perform the segmentation of all the requested images.
    if prediction_proba_activate:
        prediction, prediction_proba = apply_convnet(path_acquisitions, acquisitions_resolutions, path_model_folder,
                                                     config_dict, ckpt_name=ckpt_name,
                                                     inference_batch_size=inference_batch_size,
                                                     overlap_value=overlap_value,
                                                     resampled_resolutions=resampled_resolutions,
                                                     prediction_proba_activate=prediction_proba_activate,
                                                     gpu_per=gpu_per, verbosity_level=verbosity_level)
        # Predictions are shape of image, value = class of pixel
    else:
        prediction = apply_convnet(path_acquisitions, acquisitions_resolutions, path_model_folder, config_dict,
                                   ckpt_name=ckpt_name, inference_batch_size=inference_batch_size,
                                   overlap_value=overlap_value, resampled_resolutions=resampled_resolutions,
                                   prediction_proba_activate=prediction_proba_activate, gpu_per=gpu_per,
                                   verbosity_level=verbosity_level)
        # Predictions are shape of image, value = class of pixel

    # Final part of the function : generating the image if needed/ returning values
    if write_mode:
        for i, pred in enumerate(prediction):
            # Transform the prediction to an image
            n_classes = config_dict['n_classes']
            paint_vals = [int(255 * float(j) / (n_classes - 1)) for j in range(n_classes)]

            # Create the mask with values in range 0-255
            mask = np.zeros_like(pred)
            for j in range(n_classes):
                mask[pred == j] = paint_vals[j]
            # Then we save the image
            ads.imwrite(path_acquisitions_folders[i] / segmentations_filenames[i], mask, 'png')

            axon_prediction, myelin_prediction = get_masks(path_acquisitions_folders[i] / segmentations_filenames[i])

    if prediction_proba_activate:
        return prediction, prediction_proba
    else:
        return prediction


def ensure_list_type(elem):
    """
    Transforms the argument elem into a list if it's not already its type.
    :param elem: Element to transform into a list.
    :return: A list containing the element, or the element if it is already a list.
    """
    if type(elem) != list:
        elem = [elem]
    return elem


def load_acquisitions(path_acquisitions, acquisitions_resolutions, resampled_resolutions, verbose_mode=0):
    """
    Load and resamples acquisitions located in the indicated folders' paths.
    :param path_acquisitions: List of paths to the acquisitions images.
    :param acquisitions_resolutions: List of float containing the resolutions the acquisitions were acquired with.
    :param resampled_resolutions: List of resolutions (floats) to resample to.
    :param verbose_mode: Int, how much information to display.
    :return:
    """
    # If string, convert to Path objects
    path_acquisitions = convert_path(path_acquisitions)

    path_acquisitions, acquisitions_resolutions, resampled_resolutions = list(map(
        ensure_list_type, [path_acquisitions, acquisitions_resolutions, resampled_resolutions]))

    if verbose_mode >= 2:
        print("Loading acquisitions ...")

    # Reading acquisitions images and loading them in the RAM, with their respective acquisition resolution.
    # Then resampling the acquisitions images to the target resolution that the network uses.

    original_acquisitions, resampled_acquisitions, original_acquisitions_shapes = [], [], []

    for path_img in path_acquisitions:

        original_acquisitions.append(ads.imread(path_img))
        original_acquisitions_shapes.append(original_acquisitions[-1].shape)

    # Resampling acquisitions to the target resolution

    if verbose_mode >= 2:
        print("Rescaling acquisitions to the target resolution ...")

    resampling_coeffs = [current_acquisition_resolution / resampled_resolutions[i]
                         for i, current_acquisition_resolution in enumerate(acquisitions_resolutions)]

    for i, current_original_acquisition in enumerate(original_acquisitions):
        resampled_acquisitions.append(rescale(current_original_acquisition, resampling_coeffs[i],
                                              preserve_range=True).astype(int))

    return resampled_acquisitions, resampling_coeffs, original_acquisitions_shapes


def prepare_patches(resampled_acquisitions, patch_size, overlap_value=25):
    """
    Transform resampled acquisitions into patches. Each patch is also preprocessed during this step.
    :param resampled_acquisitions: List of acquisitions images that have been resampled
    :param patch_size: Input size of the network.
    :param overlap_value: How much overlap to include when doing the inference.
    :return: List of 512x512 patches ready to be fed to the network.
    """

    # Handle case when image is too small after resampling to target resolution of the model
    # test_patch=resampled_acquisitions[0]

    # dims = test_patch
    # height = dims[0]
    # width = dims[1]

    # print "height = ",height,"***"

    # if (height<=512) or (width<=512)
    # print " *** Image size error. The software requires an image input with size of at least 512x512 pixels in the target resolution of the model. *** "

    L_data, L_positions, L_n_patches = [], [], []

    for current_acquisition in resampled_acquisitions:
        image_init, data, positions = im2patches_overlap(current_acquisition, overlap_value, patch_size)
        L_data.append(data)
        L_positions.append(positions)
        L_n_patches.append(len(data))

    # Now we concatenate the list of patches to process them all together.
    L_data = [e for sublist in L_data for e in sublist]

    return L_data, L_n_patches, L_positions


def process_segmented_patches(predictions_list, L_n_patches, L_positions, L_original_acquisitions_shapes,
                              overlap_value, n_classes,
                              predictions_proba_list=None, prediction_proba_activate=False, verbose_mode=0):
    """
    Gathers the segmented patches into lists corresponding to each acquisition, stitches them and resamples them.
    :param predictions_list: List of all segmented patches.
    :param L_n_patches: List containing the number of patches related to each acquisition.
    :param L_positions: List of positions in the original acquisition (in the original resolution) of each patch.
    :param L_original_acquisitions_shapes: List of the shapes of the original acquisitions.
    :param overlap_value: Int, number of pixels to overlap.
    :param n_classes: Int, number of classes.
    :param predictions_proba_list: List of the prediction probabilities for all patches. Optional.
    :param prediction_proba_activate: Boolean, whether to activate or not the prediction of probabilities.
    :param verbose_mode: Int, the level of verbosity.
    :return: the reconstructed list of segmentations, as well as the list of probability maps for each acquisition,
    if requested.
    """
    patch_size = predictions_list[0].shape[0]
    L_predictions = []
    L_predictions_proba = []
    L_n_patches_cum = np.cumsum([0] + L_n_patches)

    if verbose_mode >= 2:
        print("Resampling predictions to their original size...")

    # Gathering segmented patches belonging to the same acquisition
    for i, e in enumerate(L_n_patches_cum[:-1]):
        i0 = e
        i1 = L_n_patches_cum[i + 1]
        L_predictions.append(predictions_list[i0:i1])

        if prediction_proba_activate:
            L_predictions_proba.append(predictions_proba_list[i0:i1])

    # We stitch and resample each segmented patch to reconstruct the total segmentation
    prediction_stitcheds = [patches2im_overlap(pred_list, L_positions[i], overlap_value, patch_size) for i, pred_list in
                            enumerate(L_predictions)]
    predictions = [resize(prediction_stitched, L_original_acquisitions_shapes[i]) for i, prediction_stitched in
                   enumerate(prediction_stitcheds)]
    predictions = [prediction.astype(np.uint8) for prediction in
                   predictions]  # Rescaling operation can change the value of the pixels to float.

    # Performing the same steps for the probability maps

    if prediction_proba_activate:

        # First we create an empty list that will store all processed prediction_proba
        # (meaning reshaped so that each element of the list corresponds to a predicted image,
        # each element being of shape (patch_size, patch_size, n_classes)

        predictions_proba = []

        for i, prediction_proba_list in enumerate(L_predictions_proba):
            # We generate the predict proba matrix
            tmp = np.split(np.stack(prediction_proba_list, axis=0), n_classes, axis=-1)
            predictions_proba_list = [list(map(np.squeeze, np.split(e, L_n_patches[i], axis=0))) for e in
                                      tmp]  # We now have a list (n_classes elements) of list (n_patches elements)
            # [ class0:[ patch0:[], patch1:[], ...], class1:[ patch0:[], patch1:[],... ] ... ]

            # Stitching each class
            prediction_proba_stitched = [patches2im_overlap(e, L_positions[i], overlap_value, patch_size) for j, e in
                                         enumerate(predictions_proba_list)]  # for each class, we have a list of patches

            # Stacking in order to have juste one image with a depth of 3, one for each class
            prediction_proba = np.stack(
                [resize(e, L_original_acquisitions_shapes[i]) for e in prediction_proba_stitched], axis=-1)
            predictions_proba.append(prediction_proba)

        return predictions, predictions_proba
    else:

        return predictions


def perform_batch_inference(model, tf_session, tf_prediction_op, tf_input, batch_x, size_batch, input_size, n_classes,
                            prediction_proba_activate=False):
    """
    Performs the segmentation of all the patches in the batch.
    :param tf_session: Current Tensorflow session.
    :param tf_prediction_op: Tensorflow prediction operator.
    :param batch_x: List, batch of patches to segment.
    :param size_batch: Int, size of the current batch.
    :param input_size: Int, size of a patch.
    :param n_classes: Int, number of classes.
    :param prediction_proba_activate: Boolean, whether to compute the probability maps or not.
    :return: List of segmentation of the patches, and optionally list of the probabilty maps for each patch.
    """

    batch_x = np.reshape(batch_x,(size_batch, input_size, input_size, 1))

    p = model.predict(batch_x)

    Mask = np.argmax(p, axis=3)  # Now Mask is a 256*256 mask with Mask[i,j] = pixel_class

    batch_predictions_list = [np.squeeze(e) for e in np.split(Mask, size_batch, axis=0)]

    if prediction_proba_activate:
        # Generating the probas for each element of the batch (basically changing the shape of the prediction)
        p = p.reshape(size_batch, input_size, input_size, n_classes)

        # Reshaping and adding to the preivous list (each patch is now of size (patch_size, patch_size, n_classes) )
        batch_predictions_proba_list = [np.squeeze(e) for e in np.split(p, size_batch, axis=0)]

        return batch_predictions_list, batch_predictions_proba_list

    else:
        return batch_predictions_list



