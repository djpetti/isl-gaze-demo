from dot_drawer import DotDrawer


class GazeVis(DotDrawer):
  """ Visualizes the user's gaze point. """

  def visualize_point(self, point):
    """ Visualizes the specified point.
    Args:
      point: The x and y coordinates of the point to visualize. """
    new_x, new_y = point
    self._draw_dot(new_x, new_y)

  def hide_dot(self):
    """ Hides the dot, in the case of no prediction. """
    self._hide_dot()
