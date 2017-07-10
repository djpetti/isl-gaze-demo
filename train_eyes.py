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

import keras.backend as K
from keras.models import Model
import keras.layers as layers
import keras.optimizers as optimizers

import numpy as np

import config


batch_size = 100
# How many batches to have loaded into VRAM at once.
load_batches = 8
# Shape of the input images.
image_shape = (36, 60, 3)
# Shape of the input patches.
patch_shape = (26, 56)

# How many iterations to train for.
iterations = 500

# Learning rate hyperparameters.
learning_rate = 0.01
momentum = 0.9
# Learning rate decay.
decay = learning_rate / iterations

# Where to save the network.
save_file = "eye_model.hd5"
synsets_save_file = "synsets.pkl"
# Location of the dataset files.
dataset_files = "data/daniel_myelin/dataset"
# Location of the cache files.
cache_dir = "data/daniel_myelin"


def convert_labels(labels):
  """ Convert the raw labels from the dataset into matrices that can be fed into
  the loss function.
  Args:
    labels: The labels to convert.
  Returns:
    The converted labels. """
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
  return stack

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


def build_network():
  """
  Returns:
    The built network, ready to train. """
  #input_shape = (patch_shape[0], patch_shape[1], image_shape[2])
  input_shape = (patch_shape[0], patch_shape[1], 1)
  inputs = layers.Input(shape=input_shape)

  floats = K.cast(inputs, "float32")

  noisy = layers.GaussianNoise(0)(floats)

  values = layers.Convolution2D(50, (5, 5), strides=(1, 1),
                                activation="relu")(noisy)
  values = layers.BatchNormalization()(values)

  values = layers.Convolution2D(100, (1, 1), activation="relu")(values)
  values = layers.BatchNormalization()(values)
  values = layers.Convolution2D(50, (1, 1), activation="relu")(values)
  values = layers.BatchNormalization()(values)

  values = layers.MaxPooling2D()(values)

  values = layers.Convolution2D(100, (5, 5), strides=(1, 1),
                                activation="relu")(values)
  values = layers.BatchNormalization()(values)

  values = layers.MaxPooling2D()(values)

  values = layers.Flatten()(values)

  values = layers.Dense(256, activation="relu")(values)
  values = layers.BatchNormalization()(values)
  values = layers.Dense(128, activation="relu")(values)
  values = layers.BatchNormalization()(values)
  predictions = layers.Dense(2, activation="linear")(values)

  model = Model(inputs=inputs, outputs=predictions)
  rmsprop = optimizers.RMSprop(decay=decay)
  model.compile(optimizer=rmsprop, loss=distance_metric,
                metrics=[accuracy_metric])

  model.summary()

  return model

def main():
  logger = logging.getLogger(__name__)

  model = build_network()

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
    training_data = np.dot(training_data, [0.288, 0.587, 0.114])
    training_data = np.expand_dims(training_data, 3)
    training_data = training_data.astype(np.float32)
    training_data /= np.std(training_data)
    training_labels = convert_labels(training_labels)

    # Train the model.
    history = model.fit(training_data,
                        training_labels,
                        epochs=1,
              					batch_size=batch_size)

    training_loss.append(history.history["loss"])

    if not i % 10:
      testing_data, testing_labels = data.get_test_set()
      testing_data = np.dot(testing_data, [0.288, 0.587, 0.114])
      testing_data = np.expand_dims(testing_data, 3)
      testing_data = testing_data.astype(np.float32)
      testing_data /= np.std(testing_data)
      testing_labels = convert_labels(testing_labels)
      loss, accuracy = model.evaluate(testing_data, testing_labels,
                                      batch_size=batch_size)

      print "Loss: %f, Accuracy: %f" % (loss, accuracy)
      testing_acc.append(accuracy)

      # Save the trained model.
      model.save(save_file)

  data.exit_gracefully()

  print "Saving results..."
  results_file = open("unity_eye_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()


if __name__ == "__main__":
  main()
