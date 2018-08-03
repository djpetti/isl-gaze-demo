import os
import time

import cv2


class ImageCap:
  """ Captures images of the user's face. """

  def __init__(self, save_dir, camera=-1):
    """
    Args:
      save_dir: The location to save captured data at.
      camera: The camera to read frames from. """
    self.__save_dir = save_dir
    self.__dot_num = 0

    self.__camera = cv2.VideoCapture(camera)

  def capture_for(self, seconds):
    """ Capture images for a set number of seconds.
    Args:
      seconds: The number of seconds to capture for. """
    # We're going to capture at 30 fps.
    num_frames = seconds * 30

    # Capture the frames.
    self.__frames = []
    for _ in range(0, num_frames):
      start_time = time.time()

      ret, image = self.__camera.read()
      if not ret:
        raise RuntimeError("Could not read from camera.")

      self.__frames.append(image)

      # Keep it at the right framerate.
      elapsed = time.time() - start_time
      to_wait = 1.0 / 30.0 - elapsed
      if to_wait > 0:
        time.sleep(to_wait)

  def write_frames(self, gaze_point):
    """ Write a set of captured frames to the disk.
    Args:
      gaze_point: The point at which the user was looking. """
    gaze_x, gaze_y = gaze_point
    frame_num = 0

    for frame in self.__frames:
      # Filename for the frame.
      frame_name = "%dx%d_%d_%d.jpg" % (gaze_x, gaze_y, self.__dot_num,
                                        frame_num)
      frame_path = os.path.join(self.__save_dir, frame_name)

      # Save it there.
      cv2.imwrite(frame_path, frame)

      frame_num += 1

    self.__dot_num += 1
