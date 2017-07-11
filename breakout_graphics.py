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
    left_wall_w = top_wall_h
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
