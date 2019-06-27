# -*- coding: utf-8 -*-

from pathlib import Path


from AxonDeepSeg.network_construction import *
from AxonDeepSeg.data_management.input_data import *
from AxonDeepSeg.ads_utils import convert_path
from AxonDeepSeg.config_tools import generate_config
import AxonDeepSeg.ads_utils

import keras

from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *

# Keras import
from keras.losses import categorical_crossentropy
import keras.backend.tensorflow_backend as K


K.set_session
import tensorflow as tf


def train_model(path_trainingset, path_model, config, path_model_init=None,
                save_trainable=True, gpu=None, debug_mode=False, gpu_per=1.0):
    """
    Main function. Trains a model using the configuration parameters.
    :param path_trainingset: Path to access the trainingset.
    :param path_model: Path indicating where to save the model.
    :param config: Dict, containing the configuration parameters of the network.
    :param path_model_init: Path to where the model to use for initialization is stored.
    :param save_trainable: Boolean. If True, only saves in the model variables that are trainable (evolve from gradient)
    :param gpu: String, name of the gpu to use. Prefer use of CUDA_VISIBLE_DEVICES environment variable.
    :param debug_mode: Boolean. If activated, saves more information about the distributions of
    most trainable variables, and also outputs more information.
    :param gpu_per: Float, between 0 and 1. Percentage of GPU to use.
    :return: Nothing.
    """

    # If string, convert to Path objects
    path_trainingset = convert_path(path_trainingset)
    path_model = convert_path(path_model)

    ###################################################################################################################
    ############################################## VARIABLES INITIALIZATION ###########################################
    ###################################################################################################################



    # Results and Models
    if not path_model.exists():
        path_model.mkdir(parents=True)

    # Translating useful variables from the config file.
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    image_size = config["trainingset_patchsize"]
    thresh_indices = config["thresholds"]


    # Training and Validation Path
    path_training_set = str(path_trainingset) + "/Train"
    path_validation_set = str(path_trainingset) + "/Validation"


    # List of Training Ids
    no_train_images = int(len(os.listdir(path_training_set)) / 2)
    train_ids = [str(i) for i in range(no_train_images)]

    # List of Validation Ids
    no_valid_images = int(len(os.listdir(path_validation_set)) / 2)
    valid_ids = [str(i) for i in range(no_valid_images)]

    # Loading the Training images and masks in batch
    train_gen = DataGen(train_ids, path_training_set, batch_size=no_train_images, image_size=image_size,
                        thresh_indices=thresh_indices)
    image_train_gen, mask_train_gen = train_gen.__getitem__(0)

    # Loading the Validation images and masks in batch
    valid_gen = DataGen(valid_ids, path_validation_set, batch_size=no_valid_images, image_size=image_size,
                        thresh_indices=thresh_indices)
    image_valid_gen, mask_valid_gen = valid_gen.__getitem__(0)

    ###################################################################################################################
    ############################################# DATA AUGMENTATION ##################################################
    ###################################################################################################################

    # Data dictionary to feed into image generator
    data_gen_args = dict(horizontal_flip=True,
                         # flipping()[1],
                         vertical_flip=True,
                         # preprocessing_function = elastic
                         # flipping()[0],
                         # rotation_range = random_rotation()[0],
                         # width_shift_range = shifting(patch_size, n_classes)[1],
                         # height_shift_range = shifting(patch_size, n_classes) [0],
                         # fill_mode = "constant",
                         # cval = 0
                         )

    #####################Training###################
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2018

    # Image and Mask Data Generator
    image_generator_train = image_datagen.flow(image_train_gen, y=None, seed = seed,  batch_size=batch_size, shuffle=True)
    mask_generator_train = mask_datagen.flow(mask_train_gen, y=None, seed = seed, batch_size=batch_size, shuffle=True)

    # Just zip the two generators to get a generator that provides augmented images and masks at the same time
    train_generator = zip(image_generator_train, mask_generator_train)

    ###########################Validation###########
    data_gen_args = dict()
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)


    image_generator_valid = image_datagen.flow(x=image_valid_gen, y=None,  batch_size=batch_size, seed = seed, shuffle=False)
    mask_generator_valid = mask_datagen.flow(x=mask_valid_gen, y=None,  batch_size=batch_size, seed = seed, shuffle=False)

    # Just zip the two generators to get a generator that provides augmented images and masks at the same time
    valid_generator = zip(image_generator_valid, mask_generator_valid)

    ########################### Initalizing U-Net Model ###########
    model = uconv_net(config, bn_updated_decay=None, verbose=True)

    ########################### Tensorboard for Visualization ###########
    # Name = "SEM_3c_dataset-{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    tensorboard = TensorBoard(log_dir=str(path_model))

    ########################## Training Unet Model ###########

    # Adam Optimizer for Unet
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # Compile the model with Categorical Cross Entropy loss and Adam Optimizer
    model.compile(optimizer=adam, loss="categorical_crossentropy",
                  metrics=["accuracy", dice_axon, dice_myelin])

    train_steps = len(train_ids) // batch_size
    valid_steps = len(valid_ids) // batch_size

    ########################## Use Checkpoints to save best Acuuracy and Loss ###########

    # Save the checkpoint in the /models/path_model folder
    filepath_acc = str(path_model) + "/best_acc_model.ckpt"

    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint_acc = ModelCheckpoint(filepath_acc,
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='max', period = 5)

    # Save the checkpoint in the /models/path_model folder
    filepath_loss = str(path_model) + "/best_loss_model.ckpt"


    # Keep only a single checkpoint, the best over test loss.
    checkpoint_loss = ModelCheckpoint(filepath_loss,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=True,
                                      mode='min', period = 5)



    ########################## Use Checkpoints to save best Acuuracy and Loss ###########
    model.fit_generator(train_generator, validation_data=(valid_generator), steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        epochs=epochs, callbacks=[tensorboard, checkpoint_loss, checkpoint_acc])

    ########################## Save the model after Training ###########

    model.save(str(path_model) + '/model.hdf5')


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Save Model in ckpt format
    custom_objects = {'dice_axon': dice_axon, 'dice_myelin': dice_myelin}
    model = load_model(str(path_model) + "/model.hdf5", custom_objects=custom_objects)

    sess = K.get_session()
    # Save the model to be used by TF framework
    save_path = saver.save(sess, str(path_model) + "/model.ckpt")



# Defining the Loss and  Performance Metrics

def dice_myelin(y_true, y_pred, smooth=1e-3):
    """
    Computes the pixel-wise dice myelin coefficient from the prediction tensor outputted by the network.
    :param y_pred: Tensor, the prediction outputted by the network. Shape (N,H,W,C).
    :param y_true: Tensor, the gold standard we work with. Shape (N,H,W,C).
    :return:  dice myelin coefficient for the current batch.
    """

    y_true_f = K.flatten(y_true[..., 1])
    y_pred_f = K.flatten(y_pred[..., 1])
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def dice_axon(y_true, y_pred, smooth=1e-3):
    """
    Computes the pixel-wise dice myelin coefficient from the prediction tensor outputted by the network.
    :param y_pred: Tensor, the prediction outputed by the network. Shape (N,H,W,C).
    :param y_true: Tensor, the gold standard we work with. Shape (N,H,W,C).
    :return: dice axon coefficient for the current batch.
    """

    y_true_f = K.flatten(y_true[..., 2])
    y_pred_f = K.flatten(y_pred[..., 2])
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))



# To Call the training in the terminal

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path_training", required=True, help="")
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-co", "--config_file", required=False, help="", default="~/.axondeepseg.json")
    ap.add_argument("-m_init", "--path_model_init", required=False, help="")
    ap.add_argument("-gpu", "--GPU", required=False, help="")

    args = vars(ap.parse_args())
    path_training = Path(args["path_training"])
    path_model = Path(args["path_model"])
    path_model_init = Path(args["path_model_init"])
    config_file = args["config_file"]
    gpu = args["GPU"]

    config = generate_config(config_file)

    train_model(path_training, path_model, config, path_model_init, gpu=gpu)


if __name__ == '__main__':
    main()

