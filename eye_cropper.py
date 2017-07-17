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

  def crop_image(self, image):
    """ Crops a single image.
    Args:
      image: The image to crop.
    Returns:
      The left eye cropped from the image, or None if it failed to crop it. """
    # Flip it to be compatible with other data.
    image = np.fliplr(image)

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
