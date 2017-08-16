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
iterations = 600

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


def create_bitmask_image(x1, y1, x2, y2):
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

    face_mask = create_bitmask_image(x1, y1, x2, y2)
    face_masks.append(face_mask)

  stack = np.stack(num_labels, axis=0)
  pose_stack = np.stack(poses, axis=0)
  face_stack = np.stack(face_masks, axis=0)

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


def build_network():
  """
  Returns:
    The built network, ready to train. """
  #input_shape = (patch_shape[0], patch_shape[1], image_shape[2])
  input_shape = (patch_shape[0], patch_shape[1], 1)
  inputs = layers.Input(shape=input_shape, name="main_input")

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

  # Head pose input.
  pose_input = layers.Input(shape=(3,), name="pose_input")
  pose_values = layers.Dense(100, activation="relu")(pose_input)
  pose_values = layers.BatchNormalization()(pose_values)

  pose_values = layers.Dense(50, activation="relu")(pose_values)
  pose_values = layers.BatchNormalization()(pose_values)

  pose_values = layers.Dense(50, activation="relu")(pose_values)
  pose_values = layers.BatchNormalization()(pose_values)

  # Face mask input.
  mask_input = layers.Input(shape=(25, 25), name="mask_input")

  # We have to flatten the masks before we can use them in the FF layers.
  mask_values = layers.Flatten()(mask_input)

  mask_values = layers.Dense(100, activation="relu")(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  mask_values = layers.Dense(50, activation="relu")(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  mask_values = layers.Dense(50, activation="relu")(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  values = layers.concatenate([values, pose_values, mask_values])

  values = layers.Dense(256, activation="relu")(values)
  values = layers.BatchNormalization()(values)
  values = layers.Dense(128, activation="relu")(values)
  values = layers.BatchNormalization()(values)
  predictions = layers.Dense(2, activation="linear")(values)

  model = Model(inputs=[inputs, pose_input, mask_input], outputs=predictions)
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
    training_labels, pose_data, mask_data = convert_labels(training_labels)
    #mask_data = np.zeros(mask_data.shape)

    # Train the model.
    history = model.fit([training_data, pose_data, mask_data],
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
      testing_labels, pose_data, mask_data = convert_labels(testing_labels)
      #mask_data = np.zeros(mask_data.shape)

      loss, accuracy = model.evaluate([testing_data, pose_data, mask_data],
                                      testing_labels,
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
