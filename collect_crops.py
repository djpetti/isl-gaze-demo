#!/usr/bin/python3


import os
import shutil
import sys


""" Takes cropped images and combines them in a single directory. """


# Unique identifier for copied images.
g_image_id = 0


def copy_session(session_dir, out_dir):
  """ Copies the data from one session.
  Args:
    session_dir: The session directory.
    out_dir: The directory to copy images into. """
  global g_image_id

  for image in os.listdir(session_dir):
    if not image.endswith(".jpg"):
      continue

    image_path = os.path.join(session_dir, image)
    image_name = os.path.splitext(image)[0]
    # Construct unique destination file name.
    new_path = os.path.join(out_dir, "%s_%d.jpg" % (image_name, g_image_id))
    g_image_id += 1

    shutil.copy(image_path, new_path)


def main():
  in_dir = sys.argv[1]
  out_dir = sys.argv[2]
  # Which session will become testing data.
  test_session = sys.argv[3]

  # Make training and testing directories.
  train_dir = os.path.join(out_dir, "train")
  test_dir = os.path.join(out_dir, "test")
  if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
  if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
  os.mkdir(train_dir)
  os.mkdir(test_dir)

  # Find sessions.
  found_test = False
  for session in os.listdir(in_dir):
    session_path = os.path.join(in_dir, session)
    if not os.path.isdir(session_path):
      # Extraneous item.
      continue

    if session == test_session:
      # This is our test session.
      copy_session(session_path, test_dir)
      found_test = True
    else:
      # Otherwise, this is training data.
      copy_session(session_path, train_dir)

  if not found_test:
    print("WARNING: Did not find test session.")


if __name__ == "__main__":
  main()
