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
    paddle_height = config.SCREEN_HEIGHT / 100
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
