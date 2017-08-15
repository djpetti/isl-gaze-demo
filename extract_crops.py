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

def make_pose_filename(image_file, pose, face_box, left_eye, right_eye):
  """ Adds the pose into the image filename.
  Args:
    image_file: The base name of the image.
    pose: The pose matrix for the image.
    face_box: The face bounding box.
    left_eye: The two points defining the left eye bbox.
    right_eye: The two points defining the right eye bbox. """
  pitch, yaw, roll = pose
  point1, point2 = face_box
  p1_x, p1_y = point1
  p2_x, p2_y = point2

  left_p1, left_p2 = left_eye
  left_p1_x, left_p1_y = left_p1
  left_p2_x, left_p2_y = left_p2

  right_p1, right_p2 = right_eye
  right_p1_x, right_p1_y = right_p1
  right_p2_x, right_p2_y = right_p2

  # Add it after the gaze location.
  pre_gaze, post_gaze = image_file.split("_", 1)
  new_name = "%s_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%s" % \
             (pre_gaze, pitch, yaw, roll, p1_x, p1_y, p2_x, p2_y,
              left_p1_x, left_p1_y, left_p2_x, left_p2_y, right_p1_x,
              right_p1_y, right_p2_x, right_p2_y, post_gaze)

  return new_name

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

    # Crop the face.
    face, left_eye, right_eye = cropper.crop_image(image)
    if face is None:
      # Cropping failed.
      print("WARNING: Failed to crop '%s'." % (image_file))
      continue

    # Extract the pose.
    pose = cropper.estimate_pose()
    # Extract the face bbox.
    bbox = cropper.head_box()
    # Add the them into the image name.
    image_file = make_pose_filename(image_file, pose, bbox, left_eye, right_eye)

    # Save the image.
    out_path = os.path.join(out_dir, image_file)
    cv2.imwrite(out_path, face)

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
