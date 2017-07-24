# The dimensions of the user's screen, in pixels.
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1070

# The size of a frame from the camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# The dimensions of the user's screen, in cm.
SCREEN_WIDTH_CM = 34.5
SCREEN_HEIGHT_CM = 19.5

# Model to use for prediction.
EYE_MODEL="eye_model.hd5.6"

# The minimum ratio of eye height to eye width before we consider the eye
# closed.
EYE_OPEN_RATIO = 0.1
# Minimum confidence before we ignore detections.
MIN_CONFIDENCE = 0.50

# How many gaze readings to average in the demo. A larger number increases
# smoothness but decreases responsiveness.
AVERAGE_POINTS = 10

# Number of points awarded for each row. They are in order from top to bottom.
ROW_POINTS = [7, 7, 5, 5, 3, 3, 1, 1]
# How much to speed up by each time a brick is cleared.
SPEED_INCREASE = 0.5

# Camera parameters.
class Camera:
  # How many cm the camera is offset from the center of the screen, in the x and
  # y directions. (-y is up.)
  CAMERA_POS = [0, -10.75]
  # Angular field-of-view of the camera, in radians, for the horizontal and
  # vertical dimensions.
  CAMERA_FOV = (0.941, 0.77)
  # Maximum distance user can be from the camera, in cm.
  MAX_CAMERA_DIST = 500.0
  # The focal length of the camera, in fractions of the view size.
  FOCAL_LENGTH = 0.98

# Parameters for synthetic data augmentation.
class Synthetic:
  # Average size of a user's face, in cm.
  FACE_WIDTH = 13.4
  FACE_HEIGHT = 9.75
  # Average distance from the face plane to the axis of rotation of the neck.
  FACE_VECTOR = 12.0
  # Standard deviation of noise to introduce in face dimensions.
  FACE_STDDEV = 1.0

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
