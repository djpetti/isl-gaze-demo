#!/usr/bin/python


from gaze_predictor import GazePredictor

import breakout_graphics
import collect_main
import config
import obj_canvas


""" A version of the breakout game controled by eye gaze. """


class BreakoutGame(object):
  """ Implements the breakout game. """

  def __init__(self):
    # Create predictor.
    self.__predictor = GazePredictor(config.EYE_MODEL, average_num=1)

    # Set up the graphics for the game.
    self.__setup_graphics()

  def __setup_graphics(self):
    """ Sets up the graphics for the game. """
    # Create the game canvas.
    self.__canvas = obj_canvas.Canvas(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    # Set the background color.
    self.__canvas.set_background_color(config.BreakoutColors.BG_COLOR)

    # Create graphical elements.
    self.__walls = breakout_graphics.Walls(self.__canvas)
    self.__score_box = breakout_graphics.ScoreBox(self.__canvas)
    self.__bricks = breakout_graphics.Bricks(self.__canvas)
    self.__paddle = breakout_graphics.Paddle(self.__canvas)
    self.__ball = breakout_graphics.Ball(self.__canvas)

    # Draw the graphics.
    self.__canvas.update()

  def run_iter(self):
    """ Runs a single game iteration. """
    # Get a new gaze prediction.
    gaze_x, gaze_y = self.__predictor.predict_gaze()
    if (gaze_x == -1 and gaze_y == -1):
      # The prediction failed, so we can't update.
      return

    # Convert to screen coordinates.
    gaze_x *= config.SCREEN_WIDTH
    gaze_y *= config.SCREEN_HEIGHT

    # Update paddle position.
    self.__paddle.update_position(gaze_x)

    # Update the ball animation.
    self.__ball.update()
    # Handle any collision events.
    self.__paddle.handle_collision(self.__ball)
    self.__walls.handle_collision(self.__ball)
    self.__bricks.handle_collision(self.__ball)

    # Update the canvas.
    self.__canvas.update()

  def run(self):
    """ Runs an entire game. """
    while True:
      self.run_iter()


def main():
  # Give the user a chance to get in position.
  collect_main.show_calibration()

  # Run the game.
  game = BreakoutGame()
  game.run()


if __name__ == "__main__":
  main()
