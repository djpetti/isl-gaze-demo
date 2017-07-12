# The dimensions of the user's screen, in pixels.
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1070

# Model to use for prediction.
EYE_MODEL="eye_model.hd5.3"

# The minimum ratio of eye height to eye width before we consider the eye
# closed.
EYE_OPEN_RATIO = 0.1
# Minimum confidence before we ignore detections.
MIN_CONFIDENCE = 0.50

# How many gaze readings to average in the demo. A larger number increases
# smoothness but decreases responsiveness.
AVERAGE_POINTS = 10

# Colors for breakout game.
class BreakoutColors(object):
  # Paddle color.
  PADDLE_COLOR="#c84848"
  # Color of the ball.
  BALL_COLOR="#c84848"
  # Background color.
  BG_COLOR = "#000000"
  # Static wall color.
  WALL_COLOR = "#8e8e8e"
  # Layer colors, in order, from top to bottom.
  LAYER_COLORS = ["#c84848", "#c84848", "#b47a30", "#b47a30",
                  "#a2a22a", "#a2a22a", "#48a048", "#48a048"]
