from multiprocessing import Process, Queue
import time

import cv2

from keras.models import load_model

import numpy as np

from face_tracking import landmark_detection as ld
from face_tracking import misc

from eye_cropper import EyeCropper
import image_tools
import metrics
import train_eyes


class GazePredictor(object):
  """ Handles capturing eye images, and uses the model to predict gaze. """

  def __init__(self, model_file, capture_dev=-1, average_num=5):
    """
    Args:
      model_file: The saved model to load for predictions.
      capture_dev: The camera device to capture from.
      average_num: How many images to average in our predictions. Since we run
                   on the GPU, we can basically predict extra ones for free, but
                   it does take more time to capture and pre-process them. """
    # Initialize capture and prediction processes.
    self.__prediction_process = _CnnProcess(model_file)
    self.__capture_process = _LandmarkProcess(self.__prediction_process,
                                              capture_dev=capture_dev,
                                              average_num=average_num)

  def __del__(self):
    # Make sure internal processes have terminated.
    self.__capture_process.release()
    self.__prediction_process.release()

  def predict_gaze(self):
    """ Predicts the user's gaze based on current frames.
    Returns:
      The predicted gaze point, in normalized screen units, or (-1, -1) if it
      failed to predict the gaze. """
    # Wait for new output from the predictor.
    return self.__prediction_process.get_output()

class _CnnProcess(object):
  """ Runs the CNN prediction in a separate process on the GPU, so that it can
  be handled concurrently. """

  def __init__(self, model_file):
    """
    Args:
      model_file: The file to load the predictor model from. """
    self.__model_file = model_file

    # Create the queues.
    self.__input_queue = Queue()
    self.__output_queue = Queue()

    # Fork the predictor process.
    self.__process = Process(target=_CnnProcess.predict_forever, args=(self,))
    self.__process.start()

  def release(self):
    """ Cleans up and terminates internal process. """
    self.__process.terminate()

  def predict_forever(self):
    """ Generates predictions indefinitely. """
    # Load the model we trained.
    custom = {"distance_metric": metrics.distance_metric,
              "accuracy_metric": metrics.accuracy_metric}
    self.__predictor = load_model(self.__model_file, custom_objects=custom)

    while True:
      self.__predict_once()

  def __predict_once(self):
    """ Reads an image from the input queue, processes it, and writes a
    prediction to the output queue. """
    image_batch, pose_batch, pos_batch, timestamp = self.__input_queue.get()

    if time.time() - timestamp > 0.5:
      # If it's too old, don't bother.
      return

    # Do same pre-processing.
    image_batch = image_batch.astype(np.float32)
    image_batch -= 99
    image_batch /= np.std(image_batch)

    # Generate a prediction.
    raw_preds = self.__predictor.predict([image_batch, pose_batch, pos_batch],
                                         batch_size=len(image_batch))

    # Average all the predictions to generate a final one.
    pred = np.mean(raw_preds, axis=0)

    self.__output_queue.put((pred, timestamp))

  def add_new_input(self, images, poses, boxes, timestamp):
    """ Adds a new input to be processed. Will block.
    Args:
      images: A numpy array of the input images.
      poses: A numpy array of the input head poses.
      boxes: The head boxes that were captured.
      timestamp: The time at which this image was captured. """
    self.__input_queue.put((images, poses, boxes, timestamp))

  def get_output(self):
    """ Gets an output from the prediction process. Will block.
    Returns:
      The predicted gaze point. """
    while True:
      gaze, timestamp = self.__output_queue.get()

      # Check that it's not stale.
      if time.time() - timestamp < 0.5:
        return gaze

class _LandmarkProcess(object):
  """ Captures images from the camera, and runs landmark detection in a separate
  process. """

  def __init__(self, cnn_process, capture_dev=-1, average_num=5):
    """
    Args:
      cnn_process: The _CnnProcess to send captured images to.
      capture_dev: The camera device to capture from.
      average_num: How many images to average in our predictions. Since we run
                   on the GPU, we can basically predict extra ones for free, but
                   it does take more time to capture and pre-process them. """
    self.__cnn_process = cnn_process
    self.__capture_dev = capture_dev
    self.__average_num = average_num

    # Create the queues.
    self.__input_queue = Queue()
    self.__output_queue = Queue()

    # Fork the capture process.
    self.__process = Process(target=_LandmarkProcess.capture_forever,
                             args=(self,))
    self.__process.start()

  def release(self):
    """ Cleans up and terminates internal process. """
    self.__process.terminate()

  def __capture_eye(self):
    """ Captures a single eye image.
    Returns:
      An image of the eye, or None if no suitable image could be obtained. """
    # Capture the base image.
    ret, image = self.__camera.read()
    if not ret:
      # Could not capture image.
      return None

    # Crop the left eye.
    return self.__cropper.crop_image(image)

  def capture_forever(self):
    """ Captures and crops images indefinitely. """
    # Eye cropper to use for eye detection.
    self.__cropper = EyeCropper()
    # Video capture instance to use for reading frames.
    self.__camera = cv2.VideoCapture(self.__capture_dev)

    while True:
      self.__capture_once()

  def __capture_once(self):
    """ Captures and crops a single image. It will send it to the predictor
    process when finished. """
    # Get the input images.
    image_batch = []
    pose_batch = []
    pos_batch = []
    for _ in range(0, self.__average_num):
      eye_crop = self.__capture_eye()
      if eye_crop is None:
        # We failed to get an image.
        continue

      timestamp = time.time()

      # Get the current head pose.
      pose = self.__cropper.estimate_pose()
      pose_batch.append(pose)

      # Get head position.
      pos_p1, pos_p2 = self.__cropper.head_box()
      # Produce the bitmask version.
      mask = train_eyes.create_bitmask_image(pos_p1[0], pos_p1[1], pos_p2[0],
                                             pos_p2[1])
      pos_batch.append(mask)

      # Normalize crop size.
      eye_crop = image_tools.reshape_image(eye_crop, (56, 26))
      image_batch.append(eye_crop)

    if not len(image_batch):
      # If we somehow managed to get no images, there's not a lot we can do.
      return (-1, -1)

    # Generate a prediction.
    image_batch = np.stack(image_batch)
    pose_batch = np.stack(pose_batch)
    pose_batch = pose_batch[:, :, 0]
    pos_batch = np.stack(pos_batch)
    print pose_batch

    # Send it along.
    self.__cnn_process.add_new_input(image_batch, pose_batch, pos_batch,
                                     timestamp)
