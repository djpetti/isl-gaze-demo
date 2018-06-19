#!/usr/bin/python


import argparse

from gaze_predictor import GazePredictorWithCapture
from gaze_vis import GazeVis

import collect_main


""" A simple realtime demo that moves a point around based on a user's gaze
location. """


def main():
  parser = argparse.ArgumentParser(description="Run a simple realtime demo.")
  parser.add_argument("model", help="The model file to use for inference.")
  parser.add_argument("-s", "--skip_calibration", action="store_true",
                      help="If set, don't show calibration video.")
  args = parser.parse_args()

  # Give user a chance to get in position.
  if not args.skip_calibration:
    collect_main.show_calibration()

  # Create predictor and visualizer.
  predictor = GazePredictorWithCapture(args.model)
  visualizer = GazeVis()

  window_width, window_height = visualizer.get_window_size()

  while True:
    # Get a prediction.
    pred, _, _ = predictor.predict_gaze()
    if pred is None:
      # No valid prediction. Hide the dot.
      visualizer.hide_dot()
      continue
    x_raw, y_raw = pred

    # Convert to actual pixels.
    x_pix = int(x_raw * window_width)
    y_pix = int(y_raw * window_height)
    print (x_pix, y_pix)

    # Update visualization.
    visualizer.visualize_point((x_pix, y_pix))

if __name__ == "__main__":
  main()
