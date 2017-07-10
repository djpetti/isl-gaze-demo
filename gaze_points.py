import random

import obj_canvas as oc


class GazeControl:
  """ Handles drawing a dot on-screen and capturing the user's gaze. """

  def __init__(self, window_width=1920, window_height=1080):
    """
    Args:
      window_width: The width of the window.
      window_height: The height of the window. """
    # Create the canvas.
    self.__canvas = oc.Canvas(window_width, window_height)

    # Draw the dot.
    self.__dot = oc.Circle(self.__canvas, (0, 0), 15, fill="red")
    # Randomize position.
    self.move_dot()

  def move_dot(self):
    """ Moves the dot to a new position.
    Returns:
      The coordinates of the new dot. """
    window_width, window_height = self.__canvas.get_window_size()

    # Pick a new position.
    new_x = random.randint(0, window_width)
    new_y = random.randint(0, window_height)

    # Move the dot.
    self.__dot.set_pos(new_x, new_y)

    return (new_x, new_y)
