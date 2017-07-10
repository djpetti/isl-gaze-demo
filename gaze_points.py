import random

from dot_drawer import DotDrawer


class GazeControl(DotDrawer):
  """ Handles drawing a dot on-screen and capturing the user's gaze. """

  def move_dot(self):
    """ Moves the dot to a new position.
    Returns:
      The coordinates of the new dot. """
    # Pick a new position.
    new_x = random.randint(0, self._window_width)
    new_y = random.randint(0, self._window_height)

    # Move the dot.
    self._draw_dot(new_x, new_y)

    return (new_x, new_y)
