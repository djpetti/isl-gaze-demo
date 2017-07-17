import collections
import random

import numpy as np

import config
import obj_canvas


class Paddle(object):
  """ Represents the breakout paddle. """

  def __init__(self, canvas):
    """
    Args:
      canvas: The Canvas to draw the paddle on. """
    self.__canvas = canvas

    # Old position measurements to average with.
    self.__old_positions = collections.deque()

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

    # Put it through the averaging filter.
    self.__old_positions.append(new_x)
    if len(self.__old_positions) > config.AVERAGE_POINTS:
      self.__old_positions.popleft()
    filtered_x = np.mean(self.__old_positions)

    self.__paddle.set_pos(filtered_x, y_pos)

  def handle_collision(self, ball):
    """ Handles collisions between the paddle and the ball.
    Args:
      ball: The ball that could be colliding. """
    ball.handle_collision(self.__paddle)

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

  def handle_collision(self, ball):
    """ Handles collisions between the walls and the ball.
    Args:
      ball: The ball that could be colliding. """
    # This is pretty straightforward, because it just needs to bounce when it
    # hits.
    ball.handle_collision(self.__wall_top)
    ball.handle_collision(self.__wall_left)
    ball.handle_collision(self.__wall_right)

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
      for shape in self._shapes:
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

      # Save these so we can switch them for fives.
      self._left_rect = left_rect
      self._right_rect = right_rect

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

  class Four(_Digit):
    """ A four digit. """

    def _draw_digit(self):
      """ Draws the four digit. """
      # Calculate the size for rectangular components.
      top_rect_x = self._pos_x
      top_rect_y = self._pos_y - self._scale_y * 0.3
      top_rect_w = self._scale_x
      top_rect_h = self._scale_y * 0.4

      inner_rect_x = top_rect_x
      inner_rect_y = self._pos_y - self._scale_y * 0.4
      inner_rect_w = self._scale_x * 0.6
      inner_rect_h = self._scale_y * 0.2

      bot_rect_x = self._pos_x + self._scale_x * 0.4
      bot_rect_y = self._pos_y + self._scale_y * 0.2
      bot_rect_w = self._scale_x * 0.2
      bot_rect_h = self._scale_y * 0.6

      # Create rectangles.
      top_rect = obj_canvas.Rectangle(self._canvas,
                                      (top_rect_x, top_rect_y),
                                      (top_rect_w, top_rect_h),
                                      fill=self._color,
                                      outline=self._color)
      inner_rect = obj_canvas.Rectangle(self._canvas,
                                        (inner_rect_x, inner_rect_y),
                                        (inner_rect_w, inner_rect_h),
                                        fill=self._bg_color,
                                        outline=self._bg_color)
      bot_rect = obj_canvas.Rectangle(self._canvas,
                                      (bot_rect_x, bot_rect_y),
                                      (bot_rect_w, bot_rect_h),
                                      fill=self._color,
                                      outline=self._color)

      self._shapes.extend([top_rect, inner_rect, bot_rect])

  class Five(Two):
    """ A five digit. """

    def _draw_digit(self):
      # The five is very similar to the two. We just have to flip it.
      super(ScoreBox.Five, self)._draw_digit()

      # Switch side rectangles.
      left_pos_x, left_pos_y = self._left_rect.get_pos()
      right_pos_x, right_pos_y = self._right_rect.get_pos()

      self._left_rect.set_pos(right_pos_x, left_pos_y)
      self._right_rect.set_pos(left_pos_x, right_pos_y)

  class Six(_Digit):
    """ A six digit. """

    def _draw_digit(self):
      # Calculate the size for rectangular components.
      back_rect_pos = (self._pos_x, self._pos_y)
      back_rect_size = (self._scale_x, self._scale_y)

      inner_rect_x = self._pos_x
      inner_rect_y = self._pos_y + self._scale_y * 0.2
      inner_rect_w = self._scale_x * 0.6
      inner_rect_h = self._scale_y * 0.2

      top_rect_x = self._pos_x - self._scale_x * 0.1
      top_rect_y = self._pos_y - self._scale_y * 0.2
      top_rect_w = self._scale_x * 0.8
      top_rect_h = self._scale_y * 0.2

      # Create rectangles.
      back_rect = obj_canvas.Rectangle(self._canvas, back_rect_pos,
                                       back_rect_size,
                                       fill=self._color,
                                       outline=self._color)
      inner_rect = obj_canvas.Rectangle(self._canvas,
                                        (inner_rect_x, inner_rect_y),
                                        (inner_rect_w, inner_rect_h),
                                        fill=self._bg_color,
                                        outline=self._bg_color)
      top_rect = obj_canvas.Rectangle(self._canvas,
                                      (top_rect_x, top_rect_y),
                                      (top_rect_w, top_rect_h),
                                      fill=self._bg_color,
                                      outline=self._bg_color)

      self._shapes.extend([back_rect, inner_rect, top_rect])

  class Seven(_Digit):
    """ A seven digit. """

    def _draw_digit(self):
      # Calculate the size for rectangular components.
      side_rect_x = self._pos_x + self._scale_x * 0.4
      side_rect_y = self._pos_y
      side_rect_w = self._scale_x * 0.2
      side_rect_h = self._scale_y

      top_rect_x = self._pos_x + self._scale_x * 0.1
      top_rect_y = self._pos_y - self._scale_y * 0.4
      top_rect_w = self._scale_x * 0.8
      top_rect_h = self._scale_y * 0.2

      # Create rectangles.
      side_rect = obj_canvas.Rectangle(self._canvas,
                                       (side_rect_x, side_rect_y),
                                       (side_rect_w, side_rect_h),
                                       fill=self._color,
                                       outline=self._color)
      top_rect = obj_canvas.Rectangle(self._canvas,
                                      (top_rect_x, top_rect_y),
                                      (top_rect_w, top_rect_h),
                                      fill=self._color,
                                      outline=self._color)

      self._shapes.extend([side_rect, top_rect])

  class Eight(_Digit):
    """ An eight digit. """

    def _draw_digit(self):
      # Calculate the size for rectangular components.
      back_rect_pos = (self._pos_x, self._pos_y)
      back_rect_size = (self._scale_x, self._scale_y)

      top_rect_x = self._pos_x
      top_rect_y = self._pos_y - self._scale_y * 0.2
      top_rect_w = self._scale_x * 0.6
      top_rect_h = self._scale_y * 0.2

      bot_rect_x = top_rect_x
      bot_rect_y = self._pos_y + self._scale_y * 0.2
      bot_rect_w = top_rect_w
      bot_rect_h = top_rect_h

      # Create rectangles.
      back_rect = obj_canvas.Rectangle(self._canvas, back_rect_pos,
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

  class Nine(_Digit):
    """ A nine digit. """

    def _draw_digit(self):
      # Calculate the size for rectangular components.
      back_rect_pos = (self._pos_x, self._pos_y)
      back_rect_size = (self._scale_x, self._scale_y)

      top_rect_x = self._pos_x
      top_rect_y = self._pos_y - self._scale_y * 0.2
      top_rect_w = self._scale_x * 0.6
      top_rect_h = self._scale_y * 0.2

      bot_rect_x = self._pos_x - self._scale_x * 0.1
      bot_rect_y = self._pos_y + self._scale_y * 0.2
      bot_rect_w = self._scale_x * 0.8
      bot_rect_h = self._scale_y * 0.2

      # Create rectangles.
      back_rect = obj_canvas.Rectangle(self._canvas, back_rect_pos,
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
    self.__turn_count = 3
    self.__disp_turns = None
    self.__update_turn_count()

    # Draw the score.
    self.__score = 0
    self.__ones_digit = None
    self.__tens_digit = None
    self.__hundreds_digit = None
    self.__update_score()

  def __get_digit(self, digit):
    """ A helper function that selects the correct digit class for a number.
    Args:
      digit: The number, between 0 and 9.
    Returns:
      The digit class. """
    lut = [ScoreBox.Zero, ScoreBox.One, ScoreBox.Two, ScoreBox.Three,
           ScoreBox.Four, ScoreBox.Five, ScoreBox.Six, ScoreBox.Seven,
           ScoreBox.Eight, ScoreBox.Nine]
    return lut[digit]

  def __update_turn_count(self):
    """ Update the displayed turn counter. """
    if self.__disp_turns:
      # Delete the previous number.
      self.__disp_turns.delete()

    # Calculate position and size for turn counter.
    turn_count_x = config.SCREEN_WIDTH * 0.75
    turn_count_y = config.SCREEN_HEIGHT * 0.05
    turn_count_w = config.SCREEN_WIDTH * 0.05
    turn_count_h = config.SCREEN_HEIGHT * 0.08

    # Draw it.
    digit = self.__get_digit(self.__turn_count)
    self.__disp_turns = digit(self.__canvas,
                              (turn_count_x, turn_count_y),
                              (turn_count_w, turn_count_h))

  def __update_score(self):
    """ Updates the displayed score. """
    # Calculate position and size of score digits.
    score_right_x = config.SCREEN_WIDTH * 0.62
    score_mid_x = config.SCREEN_WIDTH * 0.55
    score_left_x = config.SCREEN_WIDTH * 0.48
    score_y = config.SCREEN_HEIGHT * 0.05

    score_w = config.SCREEN_WIDTH * 0.05
    score_h = config.SCREEN_HEIGHT * 0.08

    # Draw hundreds digit.
    if self.__hundreds_digit:
      self.__hundreds_digit.delete()
    digit = self.__get_digit(self.__score / 100)
    self.__hundreds_digit = digit(self.__canvas,
                                  (score_left_x, score_y),
                                  (score_w, score_h))

    # Draw tens digit.
    if self.__tens_digit:
      self.__tens_digit.delete()
    digit = self.__get_digit((self.__score % 100) / 10)
    self.__tens_digit = digit(self.__canvas,
                              (score_mid_x, score_y),
                              (score_w, score_h))

    # Draw ones digit.
    if self.__ones_digit:
      self.__ones_digit.delete()
    digit = self.__get_digit(self.__score % 10)
    self.__ones_digit = digit(self.__canvas,
                              (score_right_x, score_y),
                              (score_w, score_h))

  def decrement_turns(self):
    """ Decrements the number of turns a user has.
    Returns:
      True if the user had a turn, False if there were none left. """
    if not self.__turn_count:
      # Out of turns.
      return False

    self.__turn_count -= 1
    self.__update_turn_count()

    return True

  def increase_score(self, amount):
    """ Increase the user's score by a given amount. """
    self.__score += amount
    self.__update_score()

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

  def handle_collision(self, ball):
    """ Detect and handle a collision between the ball and this brick.
    Args:
      ball: The ball we could be colliding with.
    Returns:
      True if there was a collision, False otherwise. """
    coll_x, coll_y = ball.handle_collision(self.__brick)
    if (coll_x and coll_y):
      # We have a collision. Remove this brick.
      self.__brick.delete()
      return True

    return False

class BrickLayer(object):
  """ Controls a layer of bricks. """

  def __init__(self, canvas, row, color):
    """
    Args:
      canvas: The canvas to draw the brick layer on.
      row: Which row the layer is, with row 0 being the top.
      color: The color of the layer. """
    self.__canvas = canvas
    self.__row = row

    # Create individual bricks.
    self.__bricks = set()
    for col in range(0, 10):
      self.__bricks.add(Brick(self.__canvas, row, col, color))

  def handle_collision(self, ball):
    """ Detect and handle a collision between the ball and this layer.
    Args:
      ball: The ball we could be colliding with.
    Returns:
      The number of points that should be awarded, or zero if there was no
      collision. """
    # Check for each brick individually.
    to_remove = []
    points = 0
    for brick in self.__bricks:
      if brick.handle_collision(ball):
        # The brick was destroyed, so we need to remove it.
        to_remove.append(brick)

        # Look up the number of points we got.
        points = config.ROW_POINTS[self.__row]

    # Remove destroyed bricks.
    for brick in to_remove:
      self.__bricks.remove(brick)

    return points

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

  def handle_collision(self, ball):
    """ Detect and handle a collision between the ball and all the bricks.
    Args:
      ball: The ball we could be colliding with.
    Returns:
      The number of points that should be awarded, or 0 if there was no
      collision. """
    # Check for each layer individually.
    points = 0
    for layer in self.__layers:
      points += layer.handle_collision(ball)

    return points

class Ball(object):
  """ Creates the ball. """

  def __init__(self, canvas, speed=20):
    """
    Args:
      canvas: The canvas to draw the balls on.
      speed: Base velocity of the ball, in px/s. """
    self.__canvas = canvas

    # The velocity vector of the ball.
    self.__choose_velocity()
    self.__vel_mult = speed

    # Keeps track of collision data for other objects.
    self.__collisions = {}

    # Figure out the ball size.
    self.__ball_x = config.SCREEN_WIDTH / 2
    self.__ball_y = config.SCREEN_HEIGHT * 0.6
    ball_h = config.SCREEN_HEIGHT * 0.015
    ball_w = ball_h

    # Draw the ball.
    color = config.BreakoutColors.BALL_COLOR
    self.__ball = obj_canvas.Rectangle(self.__canvas,
                                       (self.__ball_x, self.__ball_y),
                                       (ball_w, ball_h),
                                       fill=color,
                                       outline=color)

  def __animate(self):
    """ Animate the ball's motion. """
    move_x = self.__vel_x * self.__vel_mult
    move_y = self.__vel_y * self.__vel_mult

    self.__ball.move(move_x, move_y)

  def __choose_velocity(self):
    """ Chooses a random starting velocity for the ball. """
    self.__vel_x = (random.random() + 0.2) % 0.5
    self.__vel_y = 1 - self.__vel_x

  def update(self):
    """ Updates the ball's state. """
    self.__animate()

  def handle_collision(self, canvas_obj):
    """ Check for a collision between the ball and another canvas object. It
    automatically makes the ball bounce.
    Args:
      canvas_obj: The canvas object to check for a collision with.
    Returns
      A tuple of booleans. The first element indicates whether there is a
      collision in the x direction, the second indicates whether there is a
      collision in the y direction. """
    collision_x, collision_y = \
        obj_canvas.CanvasObject.check_collision(self.__ball, canvas_obj)

    # Get previous collision data.
    last_collision_x = False
    last_collision_y = False
    if canvas_obj in self.__collisions:
      last_collision_x, last_collision_y = self.__collisions[canvas_obj]
    # Update it.
    self.__collisions[canvas_obj] = (collision_x, collision_y)

    if (collision_x and collision_y):
      # Bounce the ball. We're going to bounce the direction that most recently
      # started colliding.
      if not last_collision_x:
        self.__vel_x *= -1
      if not last_collision_y:
        self.__vel_y *= -1

    return (collision_x, collision_y)

  def dropped(self):
    """ Detects whether the ball dropped.
    Returns:
      True if it did, False otherwise. """
    _, y_pos = self.__ball.get_pos()
    if y_pos > config.SCREEN_HEIGHT:
      # It dropped.
      return True

    return False

  def reset(self):
    """ Resets the ball to its starting position. """
    self.__ball.set_pos(self.__ball_x, self.__ball_y)
    # Reset velocity.
    self.__choose_velocity()

  def increase_speed(self):
    """ Increases speed of the ball as the game progresses. """
    self.__vel_mult += config.SPEED_INCREASE
