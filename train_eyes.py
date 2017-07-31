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

# How many iterations to train for.
iterations = 600

# Learning rate hyperparameters.
learning_rate = 0.001
momentum = 0.9
# Learning rate decay.
decay = learning_rate / iterations
#decay = 0

# L2 regularization.
l2 = 0

# Where to save the network.
save_file = "eye_model.hd5"
daniel_save_file = "eye_model_daniel.hd5"
synsets_save_file = "synsets.pkl"
# Location of the dataset files.
dataset_files = "/training_data/isl_myelin_no_cross/dataset"
# Location of the cache files.
cache_dir = "/training_data/isl_myelin_no_cross"


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


def convert_labels(labels):
  """ Convert the raw labels from the dataset into matrices that can be fed into
  the loss function.
  Args:
    labels: The labels to convert.
  Returns:
    The converted label gaze points, poses, and face masks. """
  num_labels = []
  poses = []
  face_masks = []
  for label in labels:
    coords, pitch, yaw, roll, x1, y1, x2, y2 = label.split("_")[:8]
    x_pos, y_pos = coords.split("x")
    x_pos = float(x_pos)
    y_pos = float(y_pos)

    # Scale to screen size.
    x_pos /= config.SCREEN_WIDTH
    y_pos /= config.SCREEN_HEIGHT

    num_labels.append([x_pos, y_pos])

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
  pose_stack = np.stack(poses, axis=0)
  face_stack = np.stack(face_masks, axis=0)

  return (stack, pose_stack, face_stack)

def convert_gazecapture_labels(labels):
  """ Special label conversion function for GazeCapture dataset, which lacks
  auxilliary information.
  Args:
    labels: The labels to convert.
  Returns:
    The converted label gaze points, and zeros for everything else. """
  num_labels = []
  for label in labels:
    coords = label.split("_")[0]
    x_pos, y_pos = coords.split("x")
    x_pos = float(x_pos)
    y_pos = float(y_pos)

    # Scale to screen size.
    x_pos /= config.SCREEN_WIDTH
    y_pos /= config.SCREEN_HEIGHT

    num_labels.append([x_pos, y_pos])

  stack = np.stack(num_labels, axis=0)
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
  screen_size = K.constant([config.SCREEN_WIDTH, config.SCREEN_HEIGHT])
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
    The network-in-network layer output. """
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

  return norm3


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

  nin_path = nin_layer(50, (5, 5), noisy, padding="same")
  resid = layers.Conv2D(50, (1, 1),
                        kernel_initializer=deep_xavier())(noisy)
  mod1 = layers.add([nin_path, resid])

  values = layers.MaxPooling2D()(mod1)

  nin_path2 = nin_layer(100, (5, 5), values, padding="same")
  resid2 = layers.Conv2D(100, (1, 1),
                         kernel_initializer=deep_xavier())(values)
  mod2 = layers.add([nin_path2, resid2])

  nin_path5 = nin_layer(150, (5, 5), mod2, padding="same")
  resid5 = layers.Conv2D(150, (1, 1),
                         kernel_initializer=deep_xavier())(mod2)
  mod5 = layers.add([nin_path5, resid5])

  nin_path3 = nin_layer(200, (5, 5), mod5)
  resid3 = layers.Conv2D(200, (1, 1),
                         kernel_initializer=deep_xavier())(mod5)
  crop = layers.Cropping2D((2, 2))(resid3)
  mod3 = layers.add([nin_path3, crop])

  pool2 = layers.MaxPooling2D()(mod3)

  nin_path4 = nin_layer(200, (3, 3), pool2, padding="same")
  mod4 = layers.add([nin_path4, pool2])

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
  rmsprop = optimizers.RMSprop(lr=learning_rate, decay=decay)
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

  model = load_model(save_file, custom_objects=custom)

  # Freeze the top few layers.
  names = set(["pretr1", "pretr2", "pretr3",
               "pretr4", "pretr5", "pretr6",
               "conv2d_1", "batch_normalization_1"])
  for layer in model.layers:
    print layer.name
    if layer.name in names:
      layer.trainable = False

  model.summary()

  # Recompile with an updated optimizer.
  rmsprop = optimizers.RMSprop(lr=learning_rate, decay=decay)
  model.compile(optimizer=rmsprop, loss=distance_metric,
                metrics=[accuracy_metric])

  return model

def main():
  logger = logging.getLogger(__name__)

  model = build_network()
  #model = load_pretrained()

  data = data_loader.DataManagerLoader(batch_size, load_batches, image_shape,
                                       cache_dir, dataset_files,
                                       patch_shape=patch_shape,
                                       pca_stddev=50,
                                       patch_flip=False,
                                       raw_labels=True)

  if os.path.exists(synsets_save_file):
    logger.info("Loading existing synsets...")
    data.load(synsets_save_file)


  training_acc = []
  training_loss = []
  testing_acc = []

  for i in range(0, iterations):
    # Get a new chunk of training data.
    training_data, training_labels = data.get_train_set()
    # Convert to gray.
    training_labels, pose_data, mask_data = \
        convert_labels(training_labels)

    # Train the model.
    history = model.fit([training_data, pose_data, mask_data],
                        training_labels,
                        epochs=1,
              					batch_size=batch_size)

    training_loss.append(history.history["loss"])

    if not i % 10:
      testing_data, testing_labels = data.get_test_set()
      testing_labels, pose_data, mask_data = \
          convert_labels(testing_labels)

      loss, accuracy = model.evaluate([testing_data, pose_data, mask_data],
                                      testing_labels,
                                      batch_size=batch_size)

      print "Loss: %f, Accuracy: %f" % (loss, accuracy)
      testing_acc.append(accuracy)

      # Save the trained model.
      model.save(daniel_save_file)

  data.exit_gracefully()

  print "Saving results..."
  results_file = open("unity_eye_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()


if __name__ == "__main__":
  main()
