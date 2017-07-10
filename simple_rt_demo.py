#!/usr/bin/python


from gaze_predictor import GazePredictor
from gaze_vis import GazeVis

import collect_main
import config


""" A simple realtime demo that moves a point around based on a user's gaze
location. """


def main():
  # Give user a chance to get in position.
  collect_main.show_calibration()

  # Create predictor and visualizer.
  predictor = GazePredictor(config.EYE_MODEL)
  visualizer = GazeVis(window_width=config.SCREEN_WIDTH,
                       window_height=config.SCREEN_HEIGHT)

  while True:
    # Get a prediction.
    x_raw, y_raw = predictor.predict_gaze()
    if (x_raw == -1 and y_raw == -1):
      # No valid prediction. Hide the dot.
      visualizer.hide_dot()
      continue

    # Convert to actual pixels.
    x_pix = x_raw * config.SCREEN_WIDTH
    y_pix = y_raw * config.SCREEN_HEIGHT
    print (x_pix, y_pix)

    # Update visualization.
    visualizer.visualize_point((x_pix, y_pix))

if __name__ == "__main__":
  main()
