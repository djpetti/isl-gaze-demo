import obj_canvas as oc


class GazeVis(object):
  """ Visualizes the user's gaze point. """

  def __init__(self, window_width=1920, window_height=1070):
    # Create canvas window.
    self.__canvas = oc.Canvas(window_width, window_height)
    self.__dot = None

  def visualize_point(self, point):
    """ Visualizes the specified point.
    Args:
      point: The x and y coordinates of the point to visualize. """
    new_x, new_y = point
    if self.__dot is None:
      # Make a new one if we deleted it.
      self.__dot = oc.Circle(self.__canvas, (new_x, new_y), 15, fill="red")
      return

    self.__dot.set_pos(new_x, new_y)

  def hide_dot(self):
    """ Hides the dot, in the case of no prediction. """
    if self.__dot is None:
      # Already hidden.
      return

    self.__dot.delete()
    self.__dot = None
