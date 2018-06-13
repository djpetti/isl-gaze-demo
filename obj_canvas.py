import Tkinter as tk


class Canvas(object):
  """ Simple wrapper around Tkinter canvas. """

  def __init__(self, window_width=None, window_height=None):
    """
    Args:
      window_width: The width of the window.
      window_height: The height of the window. """
    self.__window = tk.Tk()

    if window_width is None:
      # Use the full screen width.
      self.__window_width = self.__window.winfo_screenwidth()
    if window_height is None:
      # Use the full screen height.
      self.__window_height = self.__window.winfo_screenheight()

    self.__canvas = tk.Canvas(self.__window, width=self.__window_width,
                              height=self.__window_height)
    self.__canvas.pack()

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

  def set_background_color(self, color):
    """ Sets the background color of the canvas.
    Args:
      color: The color to set it to. """
    self.__canvas.config(bg=color)


class CanvasObject(object):
  """ Handles drawing an object in a Tkinter canvas window. """

  def __init__(self, canvas, pos, fill=None, outline="black"):
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
    # The object's outline color.
    self._outline = outline

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

  def _get_bbox(self):
    """ Gets the bounding box for this object. Must be implemented by the
    subclass.
    Returns:
      The bounding box, as a tuple containing the upper left corner coordinates,
      and the lower right corner coordinates. """
    raise NotImplementedError("_get_bbox() must be implemented by subclass.")

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

  def get_pos(self):
    """
    Returns:
      The position of the object. """
    return (self._pos_x, self._pos_y)

  def move(self, x_shift, y_shift):
    """ Moves an object by a certain amount. It does not update the canvas
    afterwards.
    Args:
      x_shift: How far to move in the x direction.
      y_shift: How far to move in the y direction. """
    # Update the position.
    self._pos_x += x_shift
    self._pos_y += y_shift

    self._canvas.move_object(self._reference, x_shift, y_shift)

  def delete(self):
    """ Deletes the object from the canvas. """
    self._canvas.delete_object(self._reference)

    # Indicates that the object is not present.
    self._reference = None

  @classmethod
  def check_collision(cls, obj1, obj2):
    """ Checks if there is a collision between two objects.
    Args:
      obj1: The first object.
      obj2: The second object.
    Returns:
      A tuple of booleans. The first element indicates whether there is a
      collision in the x direction, the second indicates whether there is a
      collision in the y direction. """
    # Get the bounding boxes of both objects.
    pt1_x, pt1_y, pt2_x, pt2_y = obj1._get_bbox()
    pt3_x, pt3_y, pt4_x, pt4_y = obj2._get_bbox()

    # Check for overlap.
    half_width1 = (pt2_x - pt1_x) / 2
    half_width2 = (pt4_x - pt3_x) / 2
    half_height1 = (pt2_y - pt1_y) / 2
    half_height2 = (pt4_y - pt3_y) / 2

    center1_x = pt1_x + half_width1
    center2_x = pt3_x + half_width2
    center1_y = pt1_y + half_height1
    center2_y = pt3_y + half_height2

    center_x_dist = abs(center2_x - center1_x)
    center_y_dist = abs(center2_y - center1_y)

    collision = [False, False]
    if center_x_dist <= half_width1 + half_width2:
      collision[0] = True
    if center_y_dist <= half_height1 + half_height2:
      collision[1] = True

    return collision

class Circle(CanvasObject):
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

    p1_x, p1_y, p2_x, p2_y = self._get_bbox()
    self._reference = canvas.create_oval(p1_x, p1_y, p2_x, p2_y,
                                         fill=self._fill,
                                         outline=self._outline)

  def _get_bbox(self):
    """ See superclass documentation. """
    # Calculate corner points.
    p1_x = self._pos_x - self.__radius
    p1_y = self._pos_y - self.__radius
    p2_x = self._pos_x + self.__radius
    p2_y = self._pos_y + self.__radius

    return (p1_x, p1_y, p2_x, p2_y)

class Rectangle(CanvasObject):
  """ Draws a rectangle on the canvas. """

  def __init__(self, canvas, pos, size, **kwargs):
    """
    Args:
      canvas: The Canvas to draw on.
      pos: The center position of the rectangle.
      size: The width and height of the rectangle. """
    self.__width, self.__height = size

    super(Rectangle, self).__init__(canvas, pos, **kwargs)

  def _draw_object(self):
    """ Draw the rectangle on the canvas. """
    # Get the raw canvas to draw with.
    canvas = self._canvas.get_raw_canvas()

    p1_x, p1_y, p2_x, p2_y = self._get_bbox()
    self._reference = canvas.create_rectangle(p1_x, p1_y, p2_x, p2_y,
                                              fill=self._fill,
                                              outline=self._outline)

  def _get_bbox(self):
    # Calculate corner points.
    p1_x = self._pos_x - self.__width / 2
    p1_y = self._pos_y - self.__height / 2
    p2_x = self._pos_x + self.__width / 2
    p2_y = self._pos_y + self.__height / 2

    return (p1_x, p1_y, p2_x, p2_y)
