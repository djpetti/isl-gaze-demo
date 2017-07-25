import numpy as np

from face_tracking import landmark_detection as ld
from face_tracking import misc

import config


class EyeCropper:
  """ Handles croping the eye from a series of images. """

  def __init__(self):
    # Landmark detector to use henceforth.
    self.__detector = ld.LandmarkDetection()
    self.__pose = ld.PoseEstimation()

    # The last detection flag.
    self.__detect_flag = 1
    # The last detected landmark points.
    self.__points = None

    self.__image_shape = None

  def __crop_left_eye(self, image, pts):
    """ Crops the left eye using the landmark points.
    Args:
      image: The image to crop.
      pts: The landmark points for that image.
    Returns:
      The left eye image, or None if the eye is closed. """
    # Check whether the eye is open.
    eye_width = pts[22][0] - pts[19][0]
    eye_height = pts[24][1] - pts[21][1]

    # If the ratio of height to width is too small, consider the eye closed.
    if not eye_width:
      return None
    if (eye_height / eye_width < config.EYE_OPEN_RATIO):
      return None

    return misc.crop_eye(image, pts[19], pts[22])[0]

  def __get_face_box(self, points):
    """ Quick-and-dirty face bbox estimation based on detected points.
    Args:
      points: The detected facial landmark points. """
    # These points represent the extremeties.
    left = points[0]
    right = points[9]
    top_1 = points[2]
    top_2 = points[7]
    bot = points[40]

    # Figure out extremeties.
    low_x = left[0]
    high_x = right[0]
    low_y = min(top_1[1], top_2[1])
    high_y = bot[1]

    return ((low_x, low_y), (high_x, high_y))

  def crop_image(self, image):
    """ Crops a single image.
    Args:
      image: The image to crop.
    Returns:
      The left eye cropped from the image, or None if it failed to crop it. """
    # Flip it to be compatible with other data.
    image = np.fliplr(image)
    self.__image_shape = image.shape

    confidence = 0
    if self.__detect_flag > 0:
      # We have to perform the base detection.
      self.__points, self.__detect_flag, confidence = \
          self.__detector.ffp_detect(image)
    else:
      # We can continue tracking.
      self.__points, self.__detect_flag, confidence = \
          self.__detector.ffp_track(image, self.__points)

    if confidence < config.MIN_CONFIDENCE:
      # Not a good detection.
      return None

    # Crop the left eye.
    return self.__crop_left_eye(image, self.__points)

  def estimate_pose(self):
    """ Returns the head pose estimate for the last image it cropped.
    Returns:
      A matrix with pitch, yaw, and roll. """
    return self.__pose.weakIterative_Occlusion(self.__points)

  def head_box(self):
    """ Gets the corners of the head bounding box for the last image it cropped.
    Returns:
      Two points, representing the corners of the box. They are scaled to the
      size of the image, so 0 means all the way to the left or top, and 1 means
      all the way to the right or bottom. """
    point1, point2 = self.__get_face_box(self.__points)
    p1_x, p1_y = point1
    p2_x, p2_y = point2

    # Scale to the image shape.
    image_y, image_x, _ = self.__image_shape
    p1_x /= image_x
    p2_x /= image_x
    p1_y /= image_y
    p2_y /= image_y

    return ((p1_x, p1_y), (p2_x, p2_y))
