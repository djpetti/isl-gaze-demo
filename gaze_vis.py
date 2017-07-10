import obj_canvas as oc


class GazeVis(object):
  """ Visualizes the user's gaze point. """

  def __init__(self, window_width=1920, window_height=1070):
    # Create canvas window.
    self.__canvas = oc.Canvas(window_width, window_height)

    # Draw the initial point.
    self.__dot = oc.Circle(self.__canvas, (0, 0), 15, fill="red")

  def visualize_point(self, point):
    """ Visualizes the specified point.
    Args:
      point: The x and y coordinates of the point to visualize. """
    new_x, new_y = point
    self.__dot.set_pos(new_x, new_y)

  def hide_dot(self):
    """ Hides the dot, in the case of no prediction. """
    self.__dot.delete()
