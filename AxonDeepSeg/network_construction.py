from keras.layers import *
from keras.models import *

import tensorflow as tf
import AxonDeepSeg.ads_utils

def conv_relu(x, filters, kernel_size, strides, name, activation='relu', kernel_initializer='glorot_normal',
              activate_bn=True,
              bn_decay=0.999, keep_prob=1.0):
    with tf.name_scope(name):
        with tf.name_scope("convolution"):
            if activate_bn == True:

                net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
                             kernel_initializer=kernel_initializer)(x)
                net = BatchNormalization(axis=3, momentum=1 - bn_decay)(net)
                net = Activation(activation)(net)

            else:
                net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation,
                             kernel_initializer=kernel_initializer, padding='same')(x)

        net = Dropout(1 - keep_prob)(net)

    return net


def downconv(x, filters, name, kernel_size=5, strides=2, activation='relu', kernel_initializer='glorot_normal',
             activate_bn=True, bn_decay=0.999):
    with tf.name_scope(name):
        with tf.name_scope("convolution"):
            if activate_bn == True:

                net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
                             kernel_initializer=kernel_initializer)(x)
                net = BatchNormalization(axis=3, momentum=1 - bn_decay)(net)
                net = Activation(activation)(net)

            else:

                net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation,
                             kernel_initializer=kernel_initializer, padding='same')(x)

    return net


# ------------------------ NETWORK STRUCTURE ------------------------ #


def uconv_net(training_config, bn_updated_decay=None, verbose=True):
    """
    Create the U-net.
    Input :
        x : TF object to define, ensemble des patchs des images :graph input
        config : dict : described in the header.
        image_size : int : The image size

    Output :
        The U-net.
    """

    # Load the variables
    image_size = training_config["trainingset_patchsize"]
    n_classes = training_config["n_classes"]
    depth = training_config["depth"]
    dropout = training_config["dropout"]
    number_of_convolutions_per_layer = training_config["convolution_per_layer"]
    size_of_convolutions_per_layer = training_config["size_of_convolutions_per_layer"]
    features_per_convolution = training_config["features_per_convolution"]
    downsampling = training_config["downsampling"]
    activate_bn = training_config["batch_norm_activate"]
    if bn_updated_decay is None:
        bn_decay = training_config["batch_norm_decay_starting_decay"]
    else:
        bn_decay = bn_updated_decay

    # Input picture shape is [batch_size, height, width, number_channels_in] (number_channels_in = 1 for the input layer)

    data_temp_size = [image_size]
    relu_results = []

    ####################################################################
    ######################### CONTRACTION PHASE ########################
    ####################################################################

    X = Input((image_size, image_size, 1))
    net = X


    for i in range(depth):

        for conv_number in range(number_of_convolutions_per_layer[i]):

            if verbose:
                """
                print(('Layer: ', i, ' Conv: ', conv_number, 'Features: ', features_per_convolution[i][conv_number]))
                print(('Size:', size_of_convolutions_per_layer[i][conv_number]))"""

            net = conv_relu(net, filters=features_per_convolution[i][conv_number][1],
                            kernel_size=size_of_convolutions_per_layer[i][conv_number], strides=1,
                            activation='relu', kernel_initializer='glorot_normal', activate_bn=activate_bn,
                            bn_decay=bn_decay,keep_prob=dropout, name='cconv-d' + str(i) + '-c' + str(conv_number))

        relu_results.append(net)  # We keep them for the upconvolutions

        if downsampling == 'convolution':

            net = downconv(net, filters=features_per_convolution[i][conv_number][1], kernel_size=5, strides=2,
                           activation='relu', kernel_initializer='glorot_normal', activate_bn=activate_bn,
                           bn_decay=bn_decay, name='downconv-d' + str(i))

        else:

            net = MaxPooling2D((2, 2), padding='valid', strides=2, name='downmp-d' + str(i))(net)

        data_temp_size.append(data_temp_size[-1] // 2)

    ####################################################################
    ########################## EXPANSION PHASE #########################
    ####################################################################

    for i in range(depth):
        # Upsampling
        net = UpSampling2D((2, 2))(net)

        # Convolution
        net = conv_relu(net, filters=features_per_convolution[depth - i - 1][-1][1], kernel_size=2, strides=1,
                        activation='relu', kernel_initializer='glorot_normal', activate_bn=activate_bn,
                        bn_decay=bn_decay, keep_prob=dropout, name='upconv-d' + str(depth - i - 1))

        data_temp_size.append(data_temp_size[-1] * 2)

        # concatenation (see U-net article)
        net = Concatenate(axis=3)([relu_results[depth - i - 1], net])

        # Classic convolutions
        for conv_number in range(number_of_convolutions_per_layer[depth - i - 1]):
            net = conv_relu(net, filters=features_per_convolution[depth - i - 1][conv_number][1],
                            kernel_size=size_of_convolutions_per_layer[depth - i - 1][conv_number], strides=1,
                            activation='relu', kernel_initializer='glorot_normal', activate_bn=activate_bn,
                            bn_decay=bn_decay,keep_prob=dropout, name='econv-d' + str(depth - i - 1) + '-c' + str(conv_number))

    net = Conv2D(filters=n_classes, kernel_size=1, strides=1, name='finalconv', padding='same', activation="softmax")(net)

    model = Model(inputs=X, outputs=net)

    return model
