#!/usr/bin/python

import argparse
import os
import shutil

import numpy as np
import scipy.io


def load_gaze_data(filename):
  """ Loads and curates gaze data from a matlab file.
  Args:
    filename: The file to load from.
  Returns:
    The raw gaze data, in a Numpy array. """
  mat_file = scipy.io.loadmat(filename)
  return (mat_file["gaze"])

def process_session(session_path, subject, session, all_image_data):
  """ Processes data from a single session.
  Args:
    session_path: The path to a particular head pose group in that session.
    subject: The name of the subject.
    session: The name of the session.
    all_image_data: List to write the processed data to. """
  data_path = os.path.join(session_path, "raw_image")
  if not os.path.exists(data_path):
    print "Could not find data dir: %s" % (data_path)
    return

  # Compile file names for images.
  image_numbers = []
  for image_name in os.listdir(data_path):
    if not image_name.endswith(".jpg"):
      # Extraneous.
      continue
    image_numbers.append(int(image_name.rstrip(".jpg")))

  # We need to load them in the right order to get the right gaze points.
  image_numbers.sort()

  # Include the full paths.
  image_paths = []
  for image_num in image_numbers:
    image_paths.append(os.path.join(data_path, "%05d.jpg" % (image_num)))

  # Path to the gaze data.
  gaze_path = os.path.join(data_path, "gaze.mat")
  # Load the relevant label data.
  labels = load_gaze_data(gaze_path)

  # Write everything into the main list.
  for i in range(0, len(image_paths)):
    image_path = image_paths[i]
    label = labels[i]
    all_image_data.append((image_path, label, subject, session))

def load_all_subjects(dataset_dir):
  """ Loads the data for all subjects.
  Args:
    dataset_dir: The base dataset directory.
  Returns:
    A list of tuples containing each image path, the label, the subject, and the
    session. """
  all_image_data = []

  for subject in os.listdir(dataset_dir):
    print "Loading data for %s..." % (subject)

    subject_path = os.path.join(dataset_dir, subject)

    if not os.path.isdir(subject_path):
      # Something extraneous.
      print "Ignoring extraneous subject: %s" % (subject)
      continue

    for day in os.listdir(subject_path):
      day_path = os.path.join(subject_path, day)

      if (not os.path.isdir(day_path) or not day.startswith("day")):
        print "Ignoring extraneous day: %s" % (day)
        continue

      # Look at the head positions.
      for head_pos in ["Left", "Center", "Right"]:
        head_pos_path = os.path.join(day_path, head_pos)

        process_session(head_pos_path, subject, day, all_image_data)

  return all_image_data

def copy_images(image_paths, out_dir, flatten=False):
  """ Copies the images with the correct labeled names.
  Args:
    image_data: The image data returned by load_all_subjects.
    out_dir: The base output directory.
    flatten: If true, flatten output directory structure. """
  for path, label, subject, session in image_paths:
    session_path = None
    if flatten:
      session_path = os.path.join(out_dir, "%s_%s" % (subject, session))

    else:
      subject_path = os.path.join(out_dir, subject)
      if not os.path.exists(subject_path):
        print "Adding subject: %s" % (subject)
        os.mkdir(subject_path)

      session_path = os.path.join(subject_path, session)

    if not os.path.exists(session_path):
      print "Adding session: %s" % (session)
      os.mkdir(session_path)

    # Copy the proper images.
    out_name = "%dx%d_-1_-1.jpg" % (label[0], label[1])
    out_path = os.path.join(session_path, out_name)
    shutil.copy(path, out_path)


def main():
  # Parse user arguments.
  parser = argparse.ArgumentParser( \
      description="Convert a matlab dataset to raw images.")
  parser.add_argument("dataset", help="The location of the dataset to convert.")
  parser.add_argument("output", help="Location to write output to.")
  parser.add_argument("-f", "--flatten", action="store_true",
                      help="Flatten output directory structure.")
  args = parser.parse_args()

  print "Analyzing dataset..."
  image_data = load_all_subjects(args.dataset)
  print "Copying dataset..."
  copy_images(image_data, args.output, flatten=args.flatten)
  print "Done!"

if __name__ == "__main__":
  main()
