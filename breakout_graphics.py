import config
import obj_canvas


class Paddle(object):
  """ Represents the breakout paddle. """

  def __init__(self, canvas):
    """
    Args:
      canvas: The Canvas to draw the paddle on. """
    self.__canvas = canvas

    # Figure out where the paddle will be located.
    paddle_x = config.SCREEN_WIDTH / 2
    paddle_y = config.SCREEN_HEIGHT - config.SCREEN_HEIGHT * 0.1
    paddle_width = config.SCREEN_WIDTH / 4
    paddle_height = config.SCREEN_HEIGHT * 0.015
    paddle_color = config.BreakoutColors.PADDLE_COLOR

    # The actual paddle is a rectangle.
    self.__paddle = obj_canvas.Rectangle(self.__canvas, (paddle_x, paddle_y),
                                         (paddle_width, paddle_height),
                                         fill=paddle_color,
                                         outline=paddle_color)

  def update_position(self, new_x):
    """ Updates the x position of the paddle.
    Args:
      new_x: The new x position. """
    _, y_pos = self.__paddle.get_pos()
    self.__paddle.set_pos(new_x, y_pos)

class Walls(object):
  """ This class controls the static walls. """

  def __init__(self, canvas):
    """
    Args:
      canvas: The Canvas to draw the walls on. """
    self.__canvas = canvas

    # Figure out the wall sizes.
    top_wall_x = config.SCREEN_WIDTH / 2
    top_wall_y = config.SCREEN_HEIGHT * 0.15
    top_wall_w = config.SCREEN_WIDTH - config.SCREEN_HEIGHT * 0.2
    top_wall_h = config.SCREEN_HEIGHT * 0.1

    left_wall_x = config.SCREEN_WIDTH * 0.05
    left_wall_y = config.SCREEN_HEIGHT / 2
    left_wall_w = config.SCREEN_WIDTH * 0.1
    left_wall_h = config.SCREEN_HEIGHT * 0.8

    right_wall_x = config.SCREEN_WIDTH - left_wall_x
    right_wall_y = left_wall_y
    right_wall_w = left_wall_w
    right_wall_h = left_wall_h

    wall_color = config.BreakoutColors.WALL_COLOR

    self.__wall_top = \
        obj_canvas.Rectangle(self.__canvas, (top_wall_x, top_wall_y),
                             (top_wall_w, top_wall_h),
                             fill=wall_color,
                             outline=wall_color)
    self.__wall_left = \
        obj_canvas.Rectangle(self.__canvas, (left_wall_x, left_wall_y),
                             (left_wall_w, left_wall_h),
                             fill=wall_color,
                             outline=wall_color)
    self.__wall_right = \
        obj_canvas.Rectangle(self.__canvas, (right_wall_x, right_wall_y),
                             (right_wall_w, right_wall_h),
                             fill=wall_color,
                             outline=wall_color)

class ScoreBox(object):
  """ Shows the user's score and number of turns remaining. """

  class _Digit(object):
    """ A superclass for digit objects. """

    def __init__(self, canvas, pos, scale):
      """
      Args:
        canvas: The canvas to draw on.
        pos: The center positon of the digit.
        scale: A tuple indicating the horizontal and vertical size of the digit.
      """
      self._canvas = canvas
      self._pos_x, self._pos_y = pos
      self._scale_x, self._scale_y = scale
      self._color = config.BreakoutColors.WALL_COLOR
      self._bg_color = config.BreakoutColors.BG_COLOR

      # Shapes the make up the digit.
      self._shapes = []

      # Draw the digit.
      self._draw_digit()

    def _draw_digit(self):
      """ Draws the digit. Must be overidden by subclasses. """
      raise NotImplementedError("_draw_digit must be implemented by subclass.")

    def delete(self):
      """ Delete the digit. """
      for shape in shapes:
        shape.delete()

  class Zero(_Digit):
    """ A zero digit. """

    def _draw_digit(self):
      """ Draws the zero digit. """
      # Calculate sizes for rectangular components.
      outer_rect_pos = (self._pos_x, self._pos_y)
      outer_rect_size = (self._scale_x, self._scale_y)

      inner_rect_pos = outer_rect_pos
      inner_rect_size = (self._scale_x * 0.4, self._scale_y * 0.4)

      # Create rectangles.
      outer_rect = obj_canvas.Rectangle(self._canvas, outer_rect_pos,
                                        outer_rect_size,
                                        fill=self._color,
                                        outline=self._color)
      inner_rect = obj_canvas.Rectangle(self._canvas, inner_rect_pos,
                                        inner_rect_size,
                                        fill=self._bg_color,
                                        outline=self._bg_color)

      self._shapes.extend([outer_rect, inner_rect])

  class One(_Digit):
    """ A one digit. """

    def _draw_digit(self):
      """ Draws the one digit. """
      # Calculate sizes for rectangular components.
      rect_pos = (self._pos_x, self._pos_y)
      rect_size = (self._scale_x * 0.2, self._scale_y)

      # Create rectangles.
      outer_rect = obj_canvas.Rectangle(self._canvas, rect_pos,
                                        rect_size,
                                        fill=self._color,
                                        outline=self._color)

      self._shapes.append(outer_rect)

  class Two(_Digit):
    """ A two digit. """

    def _draw_digit(self):
      """ Draws the two digit. """
      # Calculate sizes for rectangular components.
      top_rect_x = self._pos_x
      top_rect_y = self._pos_y - self._scale_y * 0.4
      top_rect_w = self._scale_x
      top_rect_h = self._scale_y * 0.2

      mid_rect_x = top_rect_x
      mid_rect_y = self._pos_y
      mid_rect_w = top_rect_w
      mid_rect_h = top_rect_h

      bot_rect_x = top_rect_x
      bot_rect_y = self._pos_y + self._scale_y * 0.4
      bot_rect_w = top_rect_w
      bot_rect_h = top_rect_h

      left_rect_x = self._pos_x - self._scale_x / 2 + top_rect_h / 2
      left_rect_y = mid_rect_y + (bot_rect_y - mid_rect_y) / 2
      left_rect_w = top_rect_h
      left_rect_h = self._scale_y / 2

      right_rect_x = self._pos_x + self._scale_x / 2 - top_rect_h / 2
      right_rect_y = top_rect_y + (mid_rect_y - top_rect_y) / 2
      right_rect_w = top_rect_h
      right_rect_h = self._scale_y / 2

      # Create rectangles.
      top_rect = obj_canvas.Rectangle(self._canvas,
                                      (top_rect_x, top_rect_y),
                                      (top_rect_w, top_rect_h),
                                      fill=self._color,
                                      outline=self._color)
      mid_rect = obj_canvas.Rectangle(self._canvas,
                                      (mid_rect_x, mid_rect_y),
                                      (mid_rect_w, mid_rect_h),
                                      fill=self._color,
                                      outline=self._color)
      bot_rect = obj_canvas.Rectangle(self._canvas,
                                      (bot_rect_x, bot_rect_y),
                                      (bot_rect_w, bot_rect_h),
                                      fill=self._color,
                                      outline=self._color)
      left_rect = obj_canvas.Rectangle(self._canvas,
                                       (left_rect_x, left_rect_y),
                                       (left_rect_w, left_rect_h),
                                       fill=self._color,
                                       outline=self._color)
      right_rect = obj_canvas.Rectangle(self._canvas,
                                        (right_rect_x, right_rect_y),
                                        (right_rect_w, right_rect_h),
                                        fill=self._color,
                                        outline=self._color)

      self._shapes.extend([top_rect, mid_rect, bot_rect, left_rect, right_rect])

  class Three(_Digit):
    """ A three digit. """

    def _draw_digit(self):
      """ Draws the three digit. """
      # Calculate the size for rectangular components.
      back_rect_pos = (self._pos_x, self._pos_y)
      back_rect_size = (self._scale_x, self._scale_y)

      top_rect_x = self._pos_x - self._scale_x * 0.1
      top_rect_y = self._pos_y - self._scale_y * 0.2
      top_rect_w = self._scale_x * 0.8
      top_rect_h = self._scale_y * 0.2

      bot_rect_x = top_rect_x
      bot_rect_y = self._pos_y + self._scale_y * 0.2
      bot_rect_w = top_rect_w
      bot_rect_h = top_rect_h

      # Create rectangles.
      back_rect = obj_canvas.Rectangle(self._canvas,
                                       back_rect_pos,
                                       back_rect_size,
                                       fill=self._color,
                                       outline=self._color)
      top_rect = obj_canvas.Rectangle(self._canvas,
                                      (top_rect_x, top_rect_y),
                                      (top_rect_w, top_rect_h),
                                      fill=self._bg_color,
                                      outline=self._bg_color)
      bot_rect = obj_canvas.Rectangle(self._canvas,
                                      (bot_rect_x, bot_rect_y),
                                      (bot_rect_w, bot_rect_h),
                                      fill=self._bg_color,
                                      outline=self._bg_color)

      self._shapes.extend([back_rect, top_rect, bot_rect])

  def __init__(self, canvas):
    """
    Args:
      canvas: The Canvas to draw the walls on. """
    self.__canvas = canvas

    # Calculate position and size for turn counter.
    turn_count_x = config.SCREEN_WIDTH * 0.75
    turn_count_y = config.SCREEN_HEIGHT * 0.05
    turn_count_w = config.SCREEN_WIDTH * 0.05
    turn_count_h = config.SCREEN_HEIGHT * 0.08

    # Draw the counter.
    self._turn_count = ScoreBox.Three(self.__canvas,
                                    (turn_count_x, turn_count_y),
                                    (turn_count_w, turn_count_h))

class Brick(object):
  """ Controls a single brick. """

  def __init__(self, canvas, row, col, color):
    """
    Args:
      canvas: The canvas to draw the brick on.
      row: Which row the brick is in, with row 0 being the top.
      col: Which column the brick is in, with col 0 being the left.
      color: The color of the brick. """
    self.__canvas = canvas

    self.__color = color

    # We should be able to fit 10 bricks between the two walls.
    col_width = config.SCREEN_WIDTH * 0.8 / 10
    # We should have 8 rows.
    row_height = config.SCREEN_HEIGHT * 0.3 / 8

    # Start positions for bricks.
    start_x = config.SCREEN_WIDTH * 0.1 + col_width / 2
    start_y = config.SCREEN_HEIGHT * 0.2 + row_height / 2

    brick_x = start_x + col_width * col
    brick_y = start_y + row_height * row
    brick_w = col_width
    brick_h = row_height

    # Draw the brick.
    self.__brick = obj_canvas.Rectangle(self.__canvas, (brick_x, brick_y),
                                        (brick_w, brick_h),
                                        fill=self.__color,
                                        outline=self.__color)

class BrickLayer(object):
  """ Controls a layer of bricks. """

  def __init__(self, canvas, row, color):
    """
    Args:
      canvas: The canvas to draw the brick layer on.
      row: Which row the layer is, with row 0 being the top.
      color: The color of the layer. """
    self.__canvas = canvas

    # Create individual bricks.
    self.__bricks = []
    for col in range(0, 10):
      self.__bricks.append(Brick(self.__canvas, row, col, color))

class Bricks(object):
  """ Creates the entire set of bricks. """

  def __init__(self, canvas):
    """
    Args:
      canvas: The canvas to draw the bricks on. """
    self.__canvas = canvas

    # Create bricks layer-by-layer.
    self.__layers = []
    for row in range(0, 8):
      # Get the color for that row.
      color = config.BreakoutColors.LAYER_COLORS[row]

      self.__layers.append(BrickLayer(self.__canvas, row, color))
