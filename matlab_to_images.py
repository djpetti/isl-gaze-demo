#!/usr/bin/python3

import os
import sys

import cv2

import numpy as np
import scipy.io


""" Helper script that converts a Matlab dataset to an image sequence. """


# Our screen size.
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080
# Original screen size.
KANG_SCREEN_WIDTH=2880
KANG_SCREEN_HEIGHT=1620


def load_mat_data(filename):
  """ Loads and curates data from a matlab file.
  Args:
    filename: The file to load from.
  Returns:
    The raw image data, in a Numpy array, and the corresponding label data. """
  mat_file = scipy.io.loadmat(filename)
  return (mat_file["eye_left"], mat_file["gaze"])

def convert_label(label, label_index):
  """ Converts a raw label to a form that we can easily parse with existing
  code.
  Args:
  label: The raw label.
    label_index: A unique number for each label.
  Returns:
    The label as a string, for use with Myelin. """
  # Convert screen sizes.
  x_pos = label[0]
  y_pos = label[1]
  x_pos = x_pos / KANG_SCREEN_WIDTH * SCREEN_WIDTH
  y_pos = y_pos / KANG_SCREEN_HEIGHT * SCREEN_HEIGHT

  return "%sx%s_%d.jpg" % (str(x_pos), str(y_pos), label_index)


def main():
  in_file = sys.argv[1]
  out_dir = sys.argv[2]

  # Load the Matlab data.
  images, labels = load_mat_data(in_file)

  # Write the output.
  for i in range(0, images.shape[0]):
    image = images[i]
    label = labels[i]

    # The image comes flipped.
    image = np.fliplr(image)
    image = np.flipud(image)

    file_name = convert_label(label, i)
    file_name = os.path.join(out_dir, file_name)

    cv2.imwrite(file_name, image)


if __name__ == "__main__":
  main()
