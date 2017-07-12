#!/usr/bin/python


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
  # Allow user to calibrate.
  show_calibration()

  # Set up the video capture.
  cap = image_cap.ImageCap(sys.argv[1])

  control = gaze_points.GazeControl(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
  gaze_point = control.move_dot()
  # Wait for the user's eye to get there.
  time.sleep(1)

  for _ in range(0, int(sys.argv[2])):
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
