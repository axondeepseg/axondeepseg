# Script storing the utility functions we are going to use in train_network.py

# 1-IMPORTS
import tensorflow as tf
import numpy as np
import collections
import AxonDeepSeg.ads_utils



# 2-FUNCTIONS DEFINITIONS
def inverted_exponential_decay(a, b, global_step, decay_period_images_seen, staircase=False):
    """
    Decays exponentially the variable between two thresholds.
    :param a: Float, inferior, starting threshold.
    :param b: Float, superior, ending threshold.
    :param global_step: Int, step we are at.
    :param decay_period_images_seen: Int, period of the decay.
    :param staircase: Boolean. Activates a staircase decay if True, continuous decay if False.
    :return: The decayed learning rate at step global_step.
    """
    if staircase:
        q, r = divmod(tf.cast(global_step, tf.int32), decay_period_images_seen)
        return a + (b - a) * (1 - tf.exp(-q))

    else:
        return a + (b - a) * (1 - tf.exp(-tf.cast(global_step, tf.float32) / decay_period_images_seen))


def poly_decay(step, initial_value, decay_period_images_seen):
    """
    Decays a variable using a polynomial law.
    :param step: number of images seen by the network since the beginning of the training.
    :param initial_value: The initial value of the variable to decay..
    :param decay_period_images_seen: the decay period in terms of images seen by the network
    (1 epoch of 10 batches of 6 images each means that 1 epoch = 60 images seen).
    Thus this value must be a multiple of the number of batches
    :return: The decayed variable.
    """

    # The magical poly decay scheduler
    # Works for every problem haha.
    factor = 1.0 - (tf.cast(step, tf.float32) / float(decay_period_images_seen))
    lrate = initial_value * np.power(factor, 0.9)

    return lrate



def generate_dict_weights(config):
    """
    Extracts the weights parameters from the configuration dictionary and rearranges them in a new dictionary which
    hierarchy is better understandable by the network.
    :param config: Dict, contains the configuration of the network.
    :return: Rearranged dict with the parameters related to weights.
    """
    weights_modifier = {}
    for key, val in list(config.items()):
        if key == 'weighted_cost_activate':
            update_recur_dict(weights_modifier, {key: val})
        elif key[:13] == 'weighted_cost':
            key1, key2 = key[13:].split('-')
            update_recur_dict(weights_modifier, {key2: val})
    return weights_modifier


def generate_dict_da(config):
    """
    Extracts the data augmentation parameters from the configuration dictionary and rearranges them in a
    new dictionary which hierarchy is better understandable by the network.
    :param config: Dict, contains the configuration of the network.
    :return: Rearranged dict with the parameters related to weights.
    """

    transformations = {}

    for key, val in list(config.items()):
        if key == 'da-type':
            key1, key2 = key.split('-')
            update_recur_dict(transformations, {key2: val})
        elif key[:3] == 'da-':
            key1, key2, key3 = key[3:].split('-')
            update_dict = {key2: {key3: val, 'order': key1}}
            update_recur_dict(transformations, update_dict)

    return transformations


def update_recur_dict(d, u):
    """
    Updates recursively a dictionary d with the values and hierarchy of dict u.
    :param d: Dictionary to update.
    :param u: Dict, update to process recursively.
    :return: Updated dict.
    """
    for k, v in list(u.items()):
        if isinstance(v, collections.Mapping):
            r = update_recur_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def count_number_parameters(trainable_variables):
    """
    Counts the number of trainable parameters (updated by the backpropagation) in the networK.
    :param trainable_variables: Tensor, output from tf.trainable_variables().
    :return: Int, number of trainable parameters in the network.
    """
    total_parameters = 0

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1

        for dim in shape:
            variable_parameters *= dim.value

        total_parameters += variable_parameters

    return total_parameters