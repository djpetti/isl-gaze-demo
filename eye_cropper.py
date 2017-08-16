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

  def __crop_eye(self, image, pts, left_index, right_index):
    """ Crops the left eye using the landmark points.
    Args:
      image: The image to crop.
      pts: The landmark points for that image.
      left_index: The index of the left eye corner.
      right_index: The index of the right eye corner.
    Returns:
      The two points defining the left eye image, or None if the eye is closed. """
    # Check whether the eye is open.
    eye_width = pts[left_index][0] - pts[right_index][0]
    eye_height = pts[left_index + 2][1] - pts[right_index + 2][1]

    # If the ratio of height to width is too small, consider the eye closed.
    if not eye_width:
      return ((None, None), (None, None))
    if (eye_height / eye_width < config.EYE_OPEN_RATIO):
      return ((None, None), (None, None))

    pt1, pt2 = misc.crop_eye(image, pts[left_index], pts[right_index])
    pt1 = np.asarray(pt1, dtype="float")
    pt2 = np.asarray(pt2, dtype="float")

    return (pt1, pt2)

  def __crop_left_eye(self, image, pts):
    """ Alias for the above function that crops just the left eye. """
    return self.__crop_eye(image, pts, 19, 22)

  def __crop_right_eye(self, image, pts):
    """ Alias for the above function that crops just the right eye. """
    return self.__crop_eye(image, pts, 25, 28)

  def crop_image(self, image):
    """ Crops a single image.
    Args:
      image: The image to crop.
    Returns:
      The face cropped from the image, plus the bounding boxes of the left and
      right eyes in fractions of the face image, or None if the detection
      failed. """
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
      return (None, (None, None), (None, None))

    # Crop the face.
    face, face_top_left = misc.crop_face(image, self.__points)
    if face is None:
      return (None, (None, None), (None, None))
    # Crop the eyes.
    left_eye_pt1, left_eye_pt2 = self.__crop_left_eye(image, self.__points)
    if left_eye_pt1[0] is None:
      return (None, (None, None), (None, None))
    right_eye_pt1, right_eye_pt2 = self.__crop_right_eye(image, self.__points)
    if right_eye_pt1[0] is None:
      return (None, (None, None), (None, None))

    # Normalize everything to the face image, because right now the points from
    # the eye crop are in terms of the full image.
    face_top_left = np.asarray(face_top_left, dtype="float")
    left_eye_pt1 = left_eye_pt1 - face_top_left
    left_eye_pt2 = left_eye_pt2 - face_top_left
    right_eye_pt1 = right_eye_pt1 - face_top_left
    right_eye_pt2 = right_eye_pt2 - face_top_left

    # Convert to frame fractions.
    face_dims = np.asarray(face.shape[:-1], dtype="float")
    left_eye_pt1 /= face_dims
    left_eye_pt2 /= face_dims
    right_eye_pt1 /= face_dims
    right_eye_pt2 /= face_dims

    # Save the face box information.
    self.__face_top_left = face_top_left
    self.__face_shape = face_dims

    return (face, (left_eye_pt1, left_eye_pt2), (right_eye_pt1, right_eye_pt2))

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
    point1 = self.__face_top_left
    point2 = self.__face_top_left + self.__face_shape

    # Scale to the image shape.
    image_shape = np.asarray(self.__image_shape[:-1], dtype="float")
    # The height and width are going to be swapped here.
    image_shape = image_shape[::-1]
    point1 /= image_shape
    point2 /= image_shape

    return (point1, point2)
