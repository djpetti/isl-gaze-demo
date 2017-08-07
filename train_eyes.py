#!/usr/bin/python


import logging


def _configure_logging():
  """ Configure logging handlers. """
  # Cofigure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("unity_gaze.log")
  file_handler.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.WARNING)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " + \
      "[%(levelname)s] %(message)s")

  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)

  root.addHandler(file_handler)
  root.addHandler(stream_handler)

# Some modules need a logger to be configured immediately.
_configure_logging()


# This forks a lot of processes, so we want to import it as soon as possible,
# when there is as little memory as possible in use.
from myelin import data_loader

from six.moves import cPickle as pickle
import argparse
import json
import os
import sys

from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
import keras.backend as K
import keras.initializers as initializers
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers

import numpy as np

import tensorflow as tf

import config

# Limit VRAM usage.
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=tf_config))

batch_size = 150
# How many batches to have loaded into VRAM at once.
load_batches = 8
# Shape of the input images.
image_shape = (36, 60, 3)
# Shape of the input patches.
patch_shape = (26, 56)

# How many iterations to train just the eye part for.
eye_iterations = 2000
# How many iterations to train the FC part for.
fc_iterations = 1200
# Iterations to pretrain with GazeCapture for.
gc_iterations = 1000

# Learning rate hyperparameters.
learning_rate = 0.005
momentum = 0.9
# Learning rate decay.
decay = learning_rate / iterations

# Hyperparameters for GazeCapture pre-training.
gc_learning_rate = 0.001
gc_decay = gc_learning_rate / iterations

# L2 regularization.
l2 = 0

# Where to save the network.
save_file = "eye_model.hd5"
gc_save_file = "eye_model_pretrained.hd5"
synsets_save_file = "synsets.pkl"
# Location of the dataset files.
dataset_files = "/training_data/isl_myelin_no_cross/dataset"
# Location of the cache files.
cache_dir = "/training_data/isl_myelin_no_cross/"
# Location of files for GazeCapture.
gc_dataset_files = "/training_data/zac_gaze/gazecapture/dataset"
gc_cache_dir = "/training_data/zac_gaze/gazecapture"


def _create_bitmask_image(x1, y1, x2, y2):
  """ Creates the bitmask image from the bbox points.
  Args:
    x1, y1: The x and y coordinates of the first point, in frame fractions.
    x2, y2: The x and y coordinates of the second point, in frame fractions.
  Returns:
    The generated bitmask image. """
  # Scale to mask size.
  x1 *= 25
  y1 *= 25
  x2 *= 25
  y2 *= 25

  x1 = int(x1)
  y1 = int(y1)
  x2 = int(x2)
  y2 = int(y2)

  # Create the interior image.
  width = x2 - x1
  height = y2 - y1
  face_box = np.ones((height, width))

  # Create the background.
  frame = np.zeros((25, 25))
  # Superimpose it correctly.
  frame[y1:y2, x1:x2] = face_box

  return frame


def convert_labels(labels, gaze_only=False):
  """ Convert the raw labels from the dataset into matrices that can be fed into
  the loss function.
  Args:
    labels: The labels to convert.
    gaze_only: If true, it will only extract the gaze point, and not the head
    pose and position.
  Returns:
    The converted label gaze points, poses, and face masks. """
  num_labels = []
  poses = []
  face_masks = []
  for label in labels:
    if not gaze_only:
      coords, pitch, yaw, roll, x1, y1, x2, y2 = label.split("_")[:8]
    else:
      coords = label.split("_")[0]

    x_pos, y_pos = coords.split("x")
    x_pos = float(x_pos)
    y_pos = float(y_pos)

    # Scale to screen size. We divide by the smaller dimension only so that we
    # end up with a 1-to-1 mapping between loss and pixel error.
    small_dim = min(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    x_pos /= small_dim
    y_pos /= small_dim

    num_labels.append([x_pos, y_pos])

    if not gaze_only:
      # Convert poses.
      pitch = float(pitch)
      yaw = float(yaw)
      roll = float(roll)

      poses.append([pitch, yaw, roll])

      # Convert bitmask.
      x1 = float(x1)
      y1 = float(y1)
      x2 = float(x2)
      y2 = float(y2)

      face_mask = _create_bitmask_image(x1, y1, x2, y2)
      face_masks.append(face_mask)

  stack = np.stack(num_labels, axis=0)
  if not gaze_only:
    pose_stack = np.stack(poses, axis=0)
    face_stack = np.stack(face_masks, axis=0)
  else:
    pose_stack = np.zeros((len(num_labels), 3))
    face_stack = np.zeros((len(num_labels), 25, 25))

  return (stack, pose_stack, face_stack)


def distance_metric(y_true, y_pred):
  """ Calculates the euclidean distance between the two labels and the
  predictions.
  Args:
    y_true: The true labels.
    y_pred: The predictions.
  Returns:
    The element-wise euclidean distance between the labels and the predictions.
  """
  diff = y_true - y_pred
  sqr = K.square(diff)
  total = K.sum(sqr, axis=1)
  return K.sqrt(total)

def accuracy_metric(y_true, y_pred):
  """ Calculates the accuracy, converting back to pixel values.
  Args:
    y_true: The true labels.
    y_pred: The predictions.
  Returns:
    The element-wise euclidean distance between the labels and the predictions.
  """
  # Scale to actual pixel values.
  small_dim = min(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
  screen_size = K.constant([small_dim, small_dim])
  y_true *= screen_size
  y_pred *= screen_size

  return distance_metric(y_true, y_pred)


def stddev_layer(layer_in):
  """ Divides the input by its standard deviation.
  Args:
    layer_in: The input tensor.
  Returns:
    The input divided by its standard deviation. """
  return layer_in / K.std(layer_in)

def bw_layer(layer_in):
  """ Converts the input to black-and-white.
  Args:
    layer_in: The input tensor.
  Returns:
    A black-and-white version of the input. """
  bw = layer_in[:, :, :, 0] * 0.288 + \
       layer_in[:, :, :, 1] * 0.587 + \
       layer_in[:, :, :, 2] * 0.114
  return K.expand_dims(bw, 3)

def nin_layer(num_filters, filter_size, top_layer, padding="valid"):
  """ Creates a network-in-network layer.
  Args:
    num_filters: The number of output filters.
    filter_size: The size of the filters.
    top_layer: The layer to build off of.
    padding: The type of padding to use.
  Returns:
    The network-in-network layer output node. """
  l2_reg = regularizers.l2

  conv = layers.Conv2D(num_filters, filter_size, kernel_regularizer=l2_reg(l2),
                       padding=padding,
                       kernel_initializer=deep_xavier())(top_layer)
  # Share across all parameters in a filter.
  act = layers.advanced_activations.PReLU(shared_axes=[1, 2])(conv)
  norm = layers.BatchNormalization()(act)

  conv2 = layers.Conv2D(num_filters * 2, (1, 1), kernel_regularizer=l2_reg(l2),
                        kernel_initializer=deep_xavier())(norm)
  act2 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(conv2)
  norm2 = layers.BatchNormalization()(act2)

  conv3 = layers.Conv2D(num_filters, (1, 1), kernel_regularizer=l2_reg(l2),
                        kernel_initializer=deep_xavier())(norm2)
  act3 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(conv3)
  norm3 = layers.BatchNormalization()(act3)
  #drop3 = layers.Dropout(0.1)(norm3)

  return norm3

def resid_module(num_filters, filter_size, top_layer, padding="valid"):
  """ Creates a residual module.
  Args:
    num_filters: The number of output filters.
    filter_size: The size of the filters.
    top_layer: The layer to build off of.
    padding: The type of padding to use.
  Returns:
    The residual module output node. """
  # Core conv layer.
  nin = nin_layer(num_filters, filter_size, top_layer, padding=padding)

  # Add a 1x1 conv if we need the sizes to sync up.
  resid = top_layer
  _, height, width, channels = K.int_shape(top_layer)
  if (channels is None or channels != num_filters):
    if not channels:
      print "WARNING: Unable to detect number of channels for top layer."

    resid = layers.Conv2D(num_filters, (1, 1))(top_layer)

  if padding == "valid":
    # In this case, we also need to crop to get the sizes to match.
    filter_w, filter_h = filter_size
    new_w = width - filter_w + 1
    new_h = height - filter_h + 1

    symmetric_w = (width - new_w) / 2
    symmetric_h = (height - new_h) / 2

    resid = layers.Cropping2D((symmetric_h, symmetric_w))(resid)

  return layers.add([nin, resid])


def deep_xavier(seed=None):
  """ This is a special variation of Glorot (normal) initialization that is
  specifically designed to aid convergence in deep ReLU-activated networks.

  It is essentially the same as standard Glorot initialization, except the
  variance is scaled by a factor of 2. In other words,
  `stddev = sqrt(2 / fan_in)`, where `fan_in` is the number of input units in
  the weight tensor.

  # Arguments
    seed: A Python integer. Used to seed the random generator.

  # Returns:
    An initializer.

  # References
    He et. al, cs.CV 2015
    https://arxiv.org/pdf/1502.01852.pdf
  """
  return initializers.VarianceScaling(scale=2.,
                                      mode='fan_in',
                                      distribution='normal',
                                      seed=seed)



def build_network():
  """
  Returns:
    The built network, ready to train. """
  #input_shape = (patch_shape[0], patch_shape[1], image_shape[2])
  input_shape = (patch_shape[0], patch_shape[1], 3)
  inputs = layers.Input(shape=input_shape, name="main_input")

  floats = K.cast(inputs, "float32")
  noisy = layers.GaussianNoise(0)(floats)

  noisy = layers.Lambda(bw_layer)(noisy)
  noisy = layers.Lambda(stddev_layer)(noisy)

  mod1 = resid_module(50, (5, 5), noisy, padding="same")

  values = layers.MaxPooling2D()(mod1)

  mod2 = resid_module(100, (5, 5), values, padding="same")
  mod5 = resid_module(150, (5, 5), mod2, padding="same")
  mod3 = resid_module(200, (5, 5), mod5)

  pool2 = layers.MaxPooling2D()(mod3)

  mod4 = resid_module(200, (3, 3), pool2, padding="same")

  # Squeeze the number of filters so the FC part isn't so huge.
  values = layers.Conv2D(50, (1, 1),
                         kernel_initializer=deep_xavier())(mod4)
  values = layers.advanced_activations.PReLU(shared_axes=[1, 2])(values)
  values = layers.BatchNormalization()(values)

  values = layers.Flatten()(values)

  values = layers.Dense(100, kernel_initializer=deep_xavier())(values)
  values = layers.advanced_activations.PReLU()(values)

  # Head pose input.
  pose_input = layers.Input(shape=(3,), name="pose_input")
  pose_values = layers.Dense(100, activation="relu",
                             kernel_initializer=deep_xavier())(pose_input)
  pose_values = layers.BatchNormalization()(pose_values)

  pose_values = layers.Dense(50, activation="relu",
                             kernel_initializer=deep_xavier())(pose_values)
  pose_values = layers.BatchNormalization()(pose_values)

  pose_values = layers.Dense(50, activation="relu",
                             kernel_initializer=deep_xavier())(pose_values)
  pose_values = layers.BatchNormalization()(pose_values)

  # Face mask input.
  mask_input = layers.Input(shape=(25, 25), name="mask_input")

  # We have to flatten the masks before we can use them in the FF layers.
  mask_values = layers.Flatten()(mask_input)

  mask_values = layers.Dense(100, activation="relu",
                             kernel_initializer=deep_xavier())(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  mask_values = layers.Dense(50, activation="relu",
                             kernel_initializer=deep_xavier())(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  mask_values = layers.Dense(50, activation="relu",
                             kernel_initializer=deep_xavier())(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  values = layers.concatenate([values, pose_values, mask_values])

  values = layers.Dense(256, activation="relu",
                        kernel_initializer=deep_xavier())(values)
  values = layers.BatchNormalization()(values)
  values = layers.Dropout(0.5)(values)
  values = layers.Dense(128, activation="relu",
                        kernel_initializer=deep_xavier())(values)
  values = layers.Dropout(0.5)(values)
  values = layers.BatchNormalization()(values)
  predictions = layers.Dense(2, activation="linear")(values)

  model = Model(inputs=[inputs, pose_input, mask_input], outputs=predictions)
  rmsprop = optimizers.RMSprop(lr=gc_learning_rate, decay=gc_decay)
  model.compile(optimizer=rmsprop, loss=distance_metric,
                metrics=[accuracy_metric])

  model.summary()

  return model

def load_pretrained():
  """ Loads a pretrained version of the network.
  Returns:
    The pretrained network. """
  # Remap metrics.
  custom = {"distance_metric": distance_metric,
            "accuracy_metric": accuracy_metric}

  model = load_model(gc_save_file, custom_objects=custom)

  # Freeze the top few layers.
  names = set(["conv2d_1", "p_re_lu_1", "batch_normalization_1",
               "conv2d_2", "p_re_lu_2", "batch_normalization_2",
               "conv2d_3", "p_re_lu_3", "batch_normalization_3",
               "conv2d_4",
               "conv2d_5", "p_re_lu_4", "batch_normalization_4",
               "conv2d_6", "p_re_lu_5", "batch_normalization_5",
               "conv2d_7", "p_re_lu_6", "batch_normalization_6",
               "conv2d_8"])
               #"conv2d_9", "p_re_lu_7", "batch_normalization_7",
               #"conv2d_10", "p_re_lu_8", "batch_normalization_8",
               #"conv2d_11", "p_re_lu_9", "batch_normalization_9",
               #"conv2d_12",
               #"conv2d_13", "p_re_lu_10", "batch_normalization_10",
               #"conv2d_14", "p_re_lu_11", "batch_normalization_11",
               #"conv2d_15", "p_re_lu_12", "batch_normalization_12",
               #"conv2d_17", "p_re_lu_13", "batch_normalization_13",
               #"conv2d_18", "p_re_lu_14", "batch_normalization_14",
               #"conv2d_19", "p_re_lu_15", "batch_normalization_15",
               #"conv2d_20", "p_re_lu_16", "batch_normalization_16"])
               #"dense_1", "p_re_lu_17"])
  for layer in model.layers:
    print layer.name
    if layer.name in names:
      layer.trainable = False

  model.summary()

  # Recompile with an updated optimizer.
  sgd = optimizers.SGD(lr=learning_rate / 5, momentum=momentum,
                       decay=decay / 5)
  model.compile(optimizer=sgd, loss=distance_metric,
                metrics=[accuracy_metric])

  return model

def train_once(model, data, use_aux=True):
  """ Run a single training iteration.
  Args:
    model: The model to train.
    data: The data loader to get data from.
    use_aux: Use auxiliary data, i.e. head pose and pos.
  Returns:
    The training loss. """
  # Get a new chunk of training data.
  training_data, training_labels = data.get_train_set()
  # Convert to a usable form.
  training_labels, pose_data, mask_data = \
      convert_labels(training_labels, gaze_only=use_aux)

  # Train the model.
  history = model.fit([training_data, pose_data, mask_data],
                      training_labels,
                      epochs=1,
                      batch_size=batch_size)

  return history.history["loss"]

def test_once(model, data, use_aux=True):
  """ Run a single testing iteration.
  Args:
    model: The model to train.
    data: The data loader to get data from.
    use_aux: Use auxiliary data, i.e. head pose and pos.
  Returns:
    The testing loss and accuracy. """
  testing_data, testing_labels = data.get_test_set()
  testing_labels, pose_data, mask_data = \
      convert_labels(testing_labels, gaze_only=use_aux)

  loss, accuracy = model.evaluate([testing_data, pose_data, mask_data],
                                   testing_labels,
                                   batch_size=batch_size)

  return loss, accuracy


def train():
  """ Trains the model. """
  logger = logging.getLogger(__name__)

  model = load_pretrained()

  data = data_loader.DataManagerLoader(batch_size, load_batches, image_shape,
                                       cache_dir, dataset_files,
                                       patch_shape=patch_shape,
                                       pca_stddev=50,
                                       patch_flip=False,
                                       raw_labels=True)

  training_acc = []
  training_loss = []
  testing_acc = []

  # Train just the eye part of the model.
  for i in range(0, eye_iterations):
    training_loss.append(train_once(model, data, use_aux=False))

    if not i % 10:
      loss, accuracy = test_once(model, data, use_aux=False)

      print "Loss: %f, Accuracy: %f" % (loss, accuracy)
      testing_acc.append(accuracy)

  print "Freezing bottom layers..."

  # Freeze the bottom layers.
  bot_layers = set(["conv2d_1", "p_re_lu_1", "batch_normalization_1",
                    "conv2d_2", "p_re_lu_2", "batch_normalization_2",
                    "conv2d_3", "p_re_lu_3", "batch_normalization_3",
                    "conv2d_4",
                    "conv2d_5", "p_re_lu_4", "batch_normalization_4",
                    "conv2d_6", "p_re_lu_5", "batch_normalization_5",
                    "conv2d_7", "p_re_lu_6", "batch_normalization_6",
                    "conv2d_8",
                    "conv2d_9", "p_re_lu_7", "batch_normalization_7",
                    "conv2d_10", "p_re_lu_8", "batch_normalization_8",
                    "conv2d_11", "p_re_lu_9", "batch_normalization_9",
                    "conv2d_12",
                    "conv2d_13", "p_re_lu_10", "batch_normalization_10",
                    "conv2d_14", "p_re_lu_11", "batch_normalization_11",
                    "conv2d_15", "p_re_lu_12", "batch_normalization_12",
                    "conv2d_17", "p_re_lu_13", "batch_normalization_13",
                    "conv2d_18", "p_re_lu_14", "batch_normalization_14",
                    "conv2d_19", "p_re_lu_15", "batch_normalization_15",
                    "conv2d_20", "p_re_lu_16", "batch_normalization_16"])

  for layer in model.layers:
    if layer.name in bot_layers:
      layer.trainable = False

  # Rebuild optimizer to reset the delay.
  sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)
  model.compile(optimizer=sgd, loss=distance_metric,
                metrics=[accuracy_metric])

  # Continue training the top ones.
  for i in range(0, fc_iterations):
    training_loss.append(train_once(model, data))

    if not i % 10:
      loss, accuracy = test_once(model, data)

      print "Loss: %f, Accuracy: %f" % (loss, accuracy)
      testing_acc.append(accuracy)

      # Save the trained model.
      model.save(save_file)

  data.exit_gracefully()

  print "Saving results..."
  results_file = open("training_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()

def train_gazecap():
  """ Pretrain on the GazeCapture dataset. """
  logger = logging.getLogger(__name__)

  print "Pre-training on GazeCapture dataset..."

  model = build_network()

  data = data_loader.DataManagerLoader(batch_size, load_batches, image_shape,
                                       gc_cache_dir, gc_dataset_files,
                                       patch_shape=patch_shape,
                                       pca_stddev=50,
                                       patch_flip=False,
                                       raw_labels=True)

  training_acc = []
  training_loss = []
  testing_acc = []

  # Train just the eye part of the model.
  for i in range(0, gc_iterations):
    training_loss.append(train_once(model, data, use_aux=False))

    if not i % 10:
      loss, accuracy = test_once(model, data, use_aux=False)

      print "Loss: %f, Accuracy: %f" % (loss, accuracy)
      testing_acc.append(accuracy)

      # Save the trained model.
      model.save(gc_save_file)

  data.exit_gracefully()

  print "Saving results..."
  results_file = open("training_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()

def main():
  parser = argparse.ArgumentParser(description="Train the eye gaze model.")
  parser.add_argument("-p", "--pre_train", action="store_true",
                      help="Whether to pretrain on gazecapture instead.")
  args = parser.parse_args()

  if args.pre_train:
    train_gazecap()
  else:
    train()

if __name__ == "__main__":
  main()
