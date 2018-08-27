#!/usr/bin/python


import argparse
import os
import random
import shutil

import cv2

import tensorflow as tf

from gaze_demo.eye_cropper import EyeCropper


def _int64_feature(value):
	""" Converts a list to an int64 feature.
	Args:
		value: The list to convert.
	Returns:
	  The corresponding feature. """
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """ Converts a list to a uint8 feature.
  Args:
    value: The list to convert.
  Returns:
    The corresponding feature. """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """ Converts a list to a float32 feature.
  Args:
    value: The list to convert.
  Returns:
    The corresponding feature. """
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Database:
  """ Writes images onto the disk. """

  def __init__(self, split_name, path):
    """
    Args:
      split_name: The name of the data split.
      path: The path to the database file. """
    self.__split = split_name
    self.__writer = tf.python_io.TFRecordWriter(path)

  def save_image(self, face_image, label_features):
    """ Saves a particular image with associated metadata to the disk.
    Args:
      face_image: The extracted face crop.
      label_features: The generated label features for this image. """
    # Compress and serialize the image.
    ret, encoded = cv2.imencode(".jpg", face_image)
    if not ret:
      print "WARNING: Encoding frame failed."
      return
    image_feature = _bytes_feature([tf.compat.as_bytes(encoded.tostring())])

    # Create the combined feature.
    dots, face_size, leye_box, reye_box, grid_box, pose = label_features
    combined_feature = {"%s/dots" % (self.__split): dots,
                        "%s/face_size" % (self.__split): face_size,
                        "%s/leye_box" % (self.__split): leye_box,
                        "%s/reye_box" % (self.__split): reye_box,
                        "%s/grid_box" % (self.__split): grid_box,
                        "%s/pose" % (self.__split): pose,
                        "%s/image" % (self.__split): image_feature}
    example = \
        tf.train.Example(features=tf.train.Features(feature=combined_feature))

    # Write it out.
    self.__writer.write(example.SerializeToString())

class ImageProcessor:
  """ Loads, processes, and writes out images to a database. """

  def __init__(self, train_db, test_db, screen_res, skip):
    """
    Args:
      train_db: The name of the training database file.
      test_db: The name of the testing database file.
      screen_res: The (x, y) resolution of the screen.
      skip: Number of early images to skip. """
    self.__screen_x, self.__screen_y = screen_res
    self.__skip_before = skip

    # The list of image paths to load.
    self.__image_paths = []

    # Eye cropper to use for extracting crops.
    self.__cropper = EyeCropper()
    # Output writers.
    self.__train_writer = Database("train", train_db)
    self.__test_writer = Database("test", test_db)

  def __is_early_image(self, image_path):
    """ Check if the image is one of the first five taken for this dot. If it
    is, the user's eyes might still be moving.
    Args:
      image_path: The path to the image. """
    # Get the file part of the path.
    filename = os.path.basename(os.path.normpath(image_path))

    # Get the frame number.
    frame_num = int(filename.split("_")[2].rstrip(".jpg"))
    if frame_num < self.__skip_before:
      return True
    return False

  def __write_image(self, image_path):
    """ Loads, processes, and saves a single image.
    Args:
      image_path: The path to the image. """
    if self.__is_early_image(image_path):
      # Skip this, because the user's eyes might still be moving.
      return

    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
      print "WARNING: Skipping invalid image: %s" % (image_path)
      return

    # Run landmark detection and find the bounding boxes.
    leye_bbox, reye_bbox, face_bbox = self.__cropper.get_bboxes(image)
    if leye_bbox is None:
      # The landmark detection failed.
      print "WARNING: Detection failed for %s." % (image_path)
      return
    # Get the face grid and estimated head pose.
    head_pose = self.__cropper.estimate_pose()
    face_grid = self.__cropper.face_grid_box()

    # Extract the dot location from the filename.
    dot_pos = self.__extract_dot_location(image_path)

    # Generate the label features.
    features = self.__generate_label_features(dot_pos, face_grid, face_bbox,
                                              leye_bbox, reye_bbox, head_pose)
    # Extract the face crop.
    face_crop = self.__extract_face_crop(image, face_bbox)
    # Combine and write out the features.
    self.__save_to_random_dataset(face_crop, features)

  def __extract_dot_location(self, image_path):
    """ Extracts the coordinates of the dot given a path to the image.
    Args:
      image_path: The path to the image.
    Returns:
      The x and y coordinates of the dot on the screen. """
    # Get the file part of the path.
    filename = os.path.basename(os.path.normpath(image_path))

    # Extract the coordinates.
    coords = filename.split("_")[0]
    x, y = coords.split("x")
    x = float(x)
    y = float(y)

    # Normalize to screen size.
    x /= self.__screen_x
    y /= self.__screen_y

    return (x, y)

  def __generate_label_features(self, dot_info, grid_info, face_info,
                                left_eye_info, right_eye_info, pose):
    """ Generates label features for data from a single image.
    Args:
      dot_info: The dot location.
      grid_info: The face grid bounding box.
      left_eye_info: The left eye bounding box.
      right_eye_info: The right eye bounding box.
      face_grid: The face grid.
      pose: The estimated head pose.
    Returns:
      Generated features, in this order: dots, face size, left eye, right eye,
      grid, pose. """
    # Crop coordinates and sizes.
    x_face, y_face, w_face, h_face = face_info
    x_leye, y_leye, w_leye, h_leye = left_eye_info
    x_reye, y_reye, w_reye, h_reye = right_eye_info
    # Face grid coordinates and sizes.
    x_grid, y_grid, w_grid, h_grid = grid_info

    # Transform eye crops so they're relative to the face instead of the entire
    # image.
    x_leye -= x_face
    x_reye -= x_face
    y_leye -= y_face
    y_reye -= y_face

    # Convert everything to frame fractions.
    x_leye /= float(w_face)
    y_leye /= float(h_face)
    w_leye /= float(w_face)
    h_leye /= float(h_face)

    x_reye /= float(w_face)
    y_reye /= float(h_face)
    w_reye /= float(w_face)
    h_reye /= float(h_face)

    x_grid /= 25.0
    y_grid /= 25.0
    w_grid /= 25.0
    h_grid /= 25.0

    # Create features.
    dots_feature = _float_feature(dot_info)
    face_size_feature = _float_feature([w_face, h_face])
    leye_box_feature = _float_feature([x_leye, y_leye, w_leye, h_leye])
    reye_box_feature = _float_feature([x_reye, y_reye, w_reye, h_reye])
    grid_box_feature = _float_feature([x_grid, y_grid, w_grid, h_grid])
    pose_feature = _float_feature(pose)

    return (dots_feature, face_size_feature, leye_box_feature, reye_box_feature,
            grid_box_feature, pose_feature)

  def __extract_face_crop(self, image, face_data):
    """ Extract the face crop from an image.
    Args:
      image: The image to process.
      face_data: The crop data for this image.
    Returns:
      A cropped version of the image. A None value in this
      list indicates a face crop that was not valid. """
    face_x, face_y, face_w, face_h = face_data

    start_x = int(face_x)
    end_x = start_x + int(face_w)
    start_y = int(face_y)
    end_y = start_y + int(face_h)

    start_x = max(0, start_x)
    end_x = min(image.shape[1], end_x)
    start_y = max(0, start_y)
    end_y = min(image.shape[0], end_y)

    # Crop the image.
    crop = image[start_y:end_y, start_x:end_x]

    # Resize the crop.
    crop = cv2.resize(crop, (400, 400))

    return crop

  def __save_to_random_dataset(self, face_image, label_features):
    """ Adds the image to a randomly-chosen dataset.
    Args:
      face_image: The image of the face to add.
      label_features: The generated label features for this image. """
    if random.random() < 0.1:
      # Save it in the testing set.
      self.__test_writer.save_image(face_image, label_features)
    else:
      # Save it in the training set.
      self.__train_writer.save_image(face_image, label_features)

  def add_images_from_dir(self, dir_path):
    """ Adds all the images from a particular directory.
    Args:
      dir_path: The path to the directory. """
    for image_file in os.listdir(dir_path):
      full_file = os.path.join(dir_path, image_file)
      self.__image_paths.append(full_file)

  def write_dbs(self):
    """ Processes the images and writes out the databases. """
    # First, randomize the order of the images.
    random.shuffle(self.__image_paths)

    num_processed = 0
    last_percentage = 0.0
    for image_path in self.__image_paths:
      # Calculate percentage complete.
      num_processed += 1
      percentage = float(num_processed) / len(self.__image_paths) * 100
      if percentage - last_percentage > 0.01:
        print "(%.2f%% complete.)" % (percentage)
        last_percentage = percentage

      self.__write_image(image_path)


def crop_sessions(in_dir, out_dir, screen_res, skip):
  """ Crops all the data in various sessions.
  Args:
    in_dir: The root directory where the session data is stored.
    out_dir: The output root directory.
    screen_res: The resolution of the user's screen.
    skip: Number of early frames to skip. """
  # Choose database files.
  train_db = os.path.join(out_dir, "eye_data_train.tfrecord")
  test_db = os.path.join(out_dir, "eye_data_test.tfrecord")
  print "Creating files: %s and %s" % (train_db, test_db)

  image_processor = ImageProcessor(train_db, test_db, screen_res, skip)

  for session in os.listdir(in_dir):
    session_path = os.path.join(in_dir, session)
    if not os.path.isdir(session_path):
      # Extraneous file.
      continue

    print "Analyzing images in %s..." % (session)
    image_processor.add_images_from_dir(session_path)

  # Write out all the databases.
  print "Processing images."
  image_processor.write_dbs()


def main():
  parser = argparse.ArgumentParser(description="Generates TFRecords datasets.")
  parser.add_argument("data_dir",
                      help="Directory containing images from each session.")
  parser.add_argument("out_dir", help="Where to write the generated datasets.")
  parser.add_argument("screen_x", type=int,
                      help="The horizontal resolution of the screen.")
  parser.add_argument("screen_y", type=int,
                      help="The vertical resolution of the screen.")
  parser.add_argument("-s", "--skip", type=int, default=5,
      help="Skip the first n frames. Useful if the eye is still moving.")
  args = parser.parse_args()

  # Crop the dataset.
  screen_res = (args.screen_x, args.screen_y)
  crop_sessions(args.data_dir, args.out_dir, screen_res, args.skip)


if __name__ == "__main__":
  main()
