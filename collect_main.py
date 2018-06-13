#!/usr/bin/python


import argparse
import sys
import time

import cv2

import config
import image_cap
import gaze_points


def show_calibration(seconds=10):
  """ Shows an image of the user to allow them to calibrate.
  Args:
    seconds: How long to show the calibration. """
  cap = cv2.VideoCapture(-1)
  start_time = time.time()

  while time.time() - start_time < seconds:
    # Capture and show the frame.
    _, img = cap.read()
    cv2.imshow("calibrate", img)
    # Update window.
    cv2.waitKey(33)

  cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser("Collect eye gaze data.")
  parser.add_argument("output_dir",
                      help="Directory in which to save the captured images.")
  parser.add_argument("num_dots", type=int,
                      help="Total number of dots to display.")
  parser.add_argument("--skip_calibration", "-s", action="store_true",
                      help="Don't show calibration window.")
  args = parser.parse_args()

  if not args.skip_calibration:
    # Allow user to calibrate.
    show_calibration()

  # Set up the video capture.
  cap = image_cap.ImageCap(args.output_dir)

  control = gaze_points.GazeControl()
  gaze_point = control.move_dot()
  # Wait for the user's eye to get there.
  time.sleep(1)

  for _ in range(0, args.num_dots):
    # Capture data for a second.
    cap.capture_for(1)

    # Move the dot.
    new_gaze_point = control.move_dot()

    # Save the data while we're waiting for the user to move their eyes.
    start_time = time.time()
    cap.write_frames(gaze_point)

    # Wait any extra.
    elapsed = time.time() - start_time
    if elapsed < 1:
      time.sleep(1 - elapsed)

    gaze_point = new_gaze_point


if __name__ == "__main__":
  main()
