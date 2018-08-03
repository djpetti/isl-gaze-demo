import network

# The dimensions of the user's screen, in pixels.
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080

# Model to use for prediction.
EYE_MODEL = "models/eye_model_both_eyes.hd5"
# Model architecture to use.
NET_ARCH = network.HeadPoseNetwork

# The minimum ratio of eye height to eye width before we consider the eye
# closed.
EYE_OPEN_RATIO = 0.1
# Minimum confidence before we ignore detections.
MIN_CONFIDENCE = 0.20
# How long a frame can be around be for we consider it stale, in seconds.
STALE_THRESHOLD = 0.5

# How many gaze readings to average in the demo. A larger number increases
# smoothness but decreases responsiveness.
AVERAGE_POINTS = 10

# Number of points awarded for each row. They are in order from top to bottom.
ROW_POINTS = [7, 7, 5, 5, 3, 3, 1, 1]
# How much to speed up by each time a brick is cleared.
SPEED_INCREASE = 0.5

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
