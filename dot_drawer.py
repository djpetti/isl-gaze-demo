from Tkinter import *


class DotDrawer:
  """ Handles drawing a dot in a Tkinter canvas window. """

  def __init__(self, window_width=1920, window_height=1070):
    """
    Args:
      window_width: The width of the window.
      window_height: The height of the window. """
    self.__window = Tk()
    self._canvas = Canvas(self.__window, width=window_width,
                          height=window_height)
    self._canvas.pack()

    self._window_width = window_width
    self._window_height = window_height

    # Create the dot.
    self.__dot = None
    self._draw_dot(0, 0)

  def _draw_dot(self, x_pos, y_pos):
    """ Draw the dot on the screen.
    Args:
      x_pos: The x-coordinate of the dot.
      y_pos: The y-coordinate of the dot. """
    if not self.__dot:
      # The dot is not yet drawn yet.
      self.__dot = self._canvas.create_oval(x_pos - 15, y_pos - 15,
                                            x_pos + 15, y_pos + 15,
                                            fill="red")

    else:
      # Otherwise, move it.
      move_x = x_pos - self.__x_pos
      move_y = y_pos - self.__y_pos

      self._canvas.move(self.__dot, move_x, move_y)

    # Update the position.
    self.__x_pos = x_pos
    self.__y_pos = y_pos

    self.__window.update()

  def _hide_dot(self):
    """ Hides the dot. """
    self._canvas.delete(self.__dot)
    self.__dot = None

    self.__window.update()
