#!/usr/bin/python


import logging


def _configure_logging():
  """ Configure logging handlers. """
  # Cofigure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("eye_gaze.log")
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

import json

from keras.backend.tensorflow_backend import set_session
import keras.optimizers as optimizers

import tensorflow as tf

import config
import metrics
import networks

# Limit VRAM usage.
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=tf_config))

batch_size = 150
# How many batches to have loaded into VRAM at once.
load_batches = 8
# Shape of the input face images.
face_image_shape = (150, 150, 3)
# Shape of the input face patches.
face_patch_shape = (144, 144)
# Shape of the input eye patches.
eye_patch_shape = (26, 56)

iterations = 500

# Learning rate hyperparameters.
learning_rate = 0.001
momentum = 0.9
# Learning rate decay.
decay = learning_rate / iterations

# L2 regularization.
l2 = 0

# Where to save the network.
save_file = "eye_model_daniel.hd5"
# Location of the dataset files.
dataset_files = "/training_data/daniel_myelin/dataset"
# Location of the cache files.
cache_dir = "/training_data/data/daniel_myelin/"


def train():
  """ Trains the model. """
  logger = logging.getLogger(__name__)

  input_shape = (eye_patch_shape[0], eye_patch_shape[1], 3)
  model = networks.build_network(input_shape, l2=l2)
  model.summary()

  # Create optimizer and compile.
  rmsprop = optimizers.RMSprop(lr=learning_rate, rho=momentum, decay=decay)
  model.compile(rmsprop, metrics.distance_metric,
                metrics=[metrics.accuracy_metric])

  data = data_loader.DataManagerLoader(batch_size, load_batches,
                                       face_image_shape,
                                       cache_dir, dataset_files,
                                       patch_shape=face_patch_shape,
                                       pca_stddev=50,
                                       patch_flip=False,
                                       raw_labels=True)
  training_acc = []
  training_loss = []
  testing_acc = []

  # Train the network.
  for i in range(0, iterations):
    training_loss.append(networks.train_once(model, data, batch_size))

    if not i % 10:
      loss, accuracy = networks.test_once(model, data, batch_size)

      print "Loss: %f, Accuracy: %f" % (loss, accuracy)
      testing_acc.append(accuracy)

      # Save the trained model.
      model.save(save_file)

  data.exit_gracefully()

  print "Saving results..."
  results_file = open("training_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()

def main():
  train()

if __name__ == "__main__":
  main()
