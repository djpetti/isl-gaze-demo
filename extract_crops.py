#!/usr/bin/python


import os
import shutil
import sys

import cv2

from eye_cropper import EyeCropper


def should_ignore(image_name):
  """ We want to ignore the first few images in each session, because the user's
  eyes may not have gotten to the dot yet.
  Args:
    image_name: The name of the image.
  Returns:
    True if we should ignore the image, false otherwise. """
  _, _, image_num = image_name.split("_")
  image_num = image_num.split(".")[0]
  image_num = int(image_num)

  return image_num < 5

def crop_dir(cropper, in_dir, out_dir):
  """ Crops an entire directory worth of images.
  Args:
    cropper: The EyeCropper instance to use.
    in_dir: The directory with the input images.
    out_dir: The directory to which the output images should be written. """
  for image_file in os.listdir(in_dir):
    if not image_file.endswith(".jpg"):
      # Not an image.
      continue

    if should_ignore(image_file):
      continue

    image = cv2.imread(os.path.join(in_dir, image_file))
    if image is None:
      # Invalid image.
      continue

    # Crop the left eye.
    left_eye = cropper.crop_image(image)
    if left_eye is None:
      # Cropping failed.
      print("WARNING: Failed to crop '%s'." % (image_file))
      continue

    # Save the image.
    out_path = os.path.join(out_dir, image_file)
    cv2.imwrite(out_path, left_eye)

def crop_sessions(in_dir, out_dir):
  """ Crops all the data in various sessions.
  Args:
    in_dir: The root directory where the session data is stored.
    out_dir: The output root directory. """
  # Initialize eye cropper.
  cropper = EyeCropper()

  for session in os.listdir(in_dir):
    session_path = os.path.join(in_dir, session)
    if not os.path.isdir(session_path):
      # Extraneous file.
      continue

    print("Cropping images in %s..." % (session))

    out_path = os.path.join(out_dir, session)
    if os.path.exists(out_path):
      print("WARNING: Removing existing session: %s" % (session))
      shutil.rmtree(out_path)
    os.mkdir(out_path)

    crop_dir(cropper, session_path, out_path)


def main():
  in_dir = sys.argv[1]
  out_dir = sys.argv[2]

  # Crop the dataset.
  crop_sessions(in_dir, out_dir)


if __name__ == "__main__":
  main()
