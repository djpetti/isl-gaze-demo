import Tkinter as tk


class Canvas(object):
  """ Simple wrapper around Tkinter canvas. """

  def __init__(self, window_width, window_height):
    """
    Args:
      window_width: The width of the window.
      window_height: The height of the window. """
    self.__window = tk.Tk()
    self.__canvas = tk.Canvas(self.__window, width=window_width,
                              height=window_height)
    self.__canvas.pack()

    self.__window_width = window_width
    self.__window_height = window_height

    self.update()

  def update(self):
    """ Updates the canvas. """
    self.__window.update()

  def move_object(self, *args, **kwargs):
    """ Shortcut for moving an object on the underlying canvas. The arguments
    are passed transparently to canvas.move. """
    self.__canvas.move(*args, **kwargs)

  def delete_object(self, *args, **kwargs):
    """ Shortcut for deleting an object from the underlying canvas. The
    arguments are passed transparently to canvas.delete. """
    self.__canvas.delete(*args, **kwargs)

  def get_raw_canvas(self):
    """ Returns: The underlying Tk canvas object. """
    return self.__canvas

  def get_window_size(self):
    """ Returns: The window width and height, as a tuple. """
    return (self.__window_width, self.__window_height)


class _CanvasObject(object):
  """ Handles drawing an object in a Tkinter canvas window. """

  def __init__(self, canvas, pos, fill=None):
    """
    Args:
      canvas: The Canvas to draw on.
      pos: The center position of the object.
      fill: The fill color of the object. """
    self._canvas = canvas

    # Keeps track of the reference for this object.
    self._reference = None
    # Keeps track of the object's center position.
    self._pos_x, self._pos_y = pos
    # The object's fill color.
    self._fill = fill

    self.__draw_object()

  def __draw_object(self):
    """ Wrapper for _draw_object that deletes an existing object first. """
    if self._reference:
      self.delete()

    self._draw_object()

  def _draw_object(self):
    """ Draws the object on the canvas. Should be implemented by the user. After
    this is called, someone still manually has to call canvas.update() to
    display it. It should also set _reference to the reference of the underlying
    canvas object, and set _pos_x and _pos_y accordingly. """
    raise NotImplementedError("_draw_object() must be implemented by subclass.")

  def set_pos(self, new_x, new_y):
    """ Moves the object to a new position.
    Args:
      new_x: The new x position.
      new_y: The new y position. """
    move_x = new_x - self._pos_x
    move_y = new_y - self._pos_y

    # Update the position.
    self._pos_x = new_x
    self._pos_y = new_y

    self._canvas.move_object(self._reference, move_x, move_y)
    self._canvas.update()

  def delete(self):
    """ Deletes the object from the canvas. """
    self._canvas.delete_object(self._reference)

    # Indicates that the object is not present.
    self._reference = None

class Circle(_CanvasObject):
  """ Draws a circle on the canvas. """

  def __init__(self, canvas, pos, radius, **kwargs):
    """
    Args:
      canvas: The Canvas to draw on.
      pos: The center position of the circle.
      radius: The radius of the circle. """
    self.__radius = radius

    super(Circle, self).__init__(canvas, pos, **kwargs)

  def _draw_object(self):
    """ Draw the circle on the screen. """
    # Get the raw canvas to draw with.
    canvas = self._canvas.get_raw_canvas()

    # Calculate corner points.
    p1_x = self._pos_x - self.__radius
    p1_y = self._pos_y - self.__radius
    p2_x = self._pos_x + self.__radius
    p2_y = self._pos_y + self.__radius

    self._reference = canvas.create_oval(p1_x, p1_y, p2_x, p2_y,
                                         fill=self._fill)
