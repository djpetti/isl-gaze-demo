from multiprocessing import Process, Queue
import logging
import time

import cv2

from keras import layers

import numpy as np

import tensorflow as tf

from eye_cropper import EyeCropper
import config


logger = logging.getLogger(__name__)


def _is_stale(timestamp):
  """ Checks if an image is stale.
  Args:
    timestamp: The timestamp of the image to check.
  Returns:
    True if the image is stale, false otherwise. """
  if time.time() - timestamp > config.STALE_THRESHOLD:
    print "Dropping stale frame."
    logger.warning("Dropping stale image from %f at %f." % \
                   (timestamp, time.time()))
    return True

  return False


class GazePredictor(object):
  """ Takes in eye images, and uses the model to predict gaze. """

  def __init__(self, model_file, display=False):
    """
    Args:
      model_file: The saved model to load for predictions.
      display: If true, it will enable a debug display that shows the image
               crops. """
    # Initialize landmark and prediction processes.
    self._prediction_process = _CnnProcess(model_file)
    self._landmark_process = _LandmarkProcess(self._prediction_process,
                                              display=display)

  def __del__(self):
    # Make sure internal processes have terminated.
    self._landmark_process.release()
    self._prediction_process.release()

  def predict_gaze(self):
    """ Predicts the user's gaze based on current frames.
    Returns:
      The predicted gaze point, in cm, the sequence number of the
      corresponding frame, and the timestamp. """
    # Wait for new output from the predictor.
    return self._prediction_process.get_output()

  def process_image(self, image, seq_num):
    """ Adds a new image to the prediction pipeline.
    Args:
      image: The image to add.
      seq_num: The sequence number of the image. """
    # Add a new timestamp for this image. This allows us to drop frames that go
    # stale.
    timestamp = time.time()

    self._landmark_process.add_new_input(image, seq_num, timestamp)

class GazePredictorWithCapture(GazePredictor):
  """ Same as a GazePredictor, except it also handles capturing images from the
  camera. """

  def __init__(self, *args, **kwargs):
    super(GazePredictorWithCapture, self).__init__(*args, **kwargs)

    # Initialize capture process.
    self._capture_process = _CaptureProcess(self._landmark_process)

  def __del__(self):
    super(GazePredictorWithCapture, self).__del__()

    self._capture_process.release()

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

  @staticmethod
  def _make_eye_pathway(eye_input):
    """ Generates the preprocessing pathway for a single eye.
    Args:
      eye_input: The raw eye input to be preprocessed.
    Returns:
      The input placeholder, and the output node. """
    # Convert to grayscale.
    eye_gray = tf.image.rgb_to_grayscale(eye_input)
    # Crop it properly.
    eye_cropped = tf.image.crop_to_bounding_box(eye_gray, 3, 3, 218, 218)
    # Resize it properly.
    eye_resized = tf.image.resize_images(eye_cropped, (32, 32),
                                          align_corners=True)
    # Normalize it.
    eye_norm = tf.map_fn(lambda frame: \
                         tf.image.per_image_standardization(frame), eye_resized)

    return eye_norm

  def release(self):
    """ Cleans up and terminates internal process. """
    self.__process.terminate()

  def predict_forever(self):
    """ Generates predictions indefinitely. """
    # Create preprocessing layer.
    eye_preproc_layer = layers.Lambda(_CnnProcess._make_eye_pathway)

    # Load the model we trained.
    model = config.NET_ARCH((224, 224, 3),
                            eye_preproc=eye_preproc_layer)
    self.__predictor = model.build()
    self.__predictor.load_weights(self.__model_file)

    while True:
      self.__predict_once()

  def __predict_once(self):
    """ Reads an image from the input queue, processes it, and writes a
    prediction to the output queue. """
    left_eye, right_eye, face, grid, pose, seq_num, timestamp = \
        self.__input_queue.get()
    if seq_num is None:
      # A None tuple means the end of the sequence. Propagate this through the
      # pipeline.
      self.__output_queue.put((None, None, None))
      return
    if left_eye is None:
      # If we have a sequence number but no images, we got a bad detection.
      self.__output_queue.put((None, seq_num, timestamp))
      return
    if _is_stale(timestamp):
      # The image is stale, so indicate that it is invalid.
      self.__output_queue.put((None, seq_num, timestamp))
      return

    # Convert everything to floats.
    left_eye = left_eye.astype(np.float32)
    right_eye = right_eye.astype(np.float32)
    face = face.astype(np.float32)

    # Add the batch dimension.
    left_eye = np.expand_dims(left_eye, axis=0)
    right_eye = np.expand_dims(right_eye, axis=0)
    face = np.expand_dims(face, axis=0)
    grid = np.expand_dims(grid, axis=0)
    pose = np.expand_dims(pose, axis=0)

    # Generate a prediction.
    pred = self.__predictor.predict([left_eye, right_eye, face, grid, pose],
                                    batch_size=1)
    # Remove the batch dimension, and convert to Python floats.
    pred = [float(x) for x in pred[0]]

    self.__output_queue.put((pred, seq_num, timestamp))

  def add_new_input(self, left_eye, right_eye, face, grid, pose, seq_num,
                    timestamp):
    """ Adds a new input to be processed. Will block.
    Args:
      left_eye: The left eye crop.
      right_eye: The right eye crop.
      face: The face crop.
      grid: The face grid.
      pose: The head pose.
      seq_num: The sequence number of the image.
      timestamp: The timestamp of the image.
    """
    self.__input_queue.put((left_eye, right_eye, face, grid, pose, seq_num,
                            timestamp))

  def get_output(self):
    """ Gets an output from the prediction process. Will block.
    Returns:
      The predicted gaze point and the sequence number. """
    return self.__output_queue.get()

class _LandmarkProcess(object):
  """ Reads images from a queue, and runs landmark detection in a separate
  process. """

  def __init__(self, cnn_process, display=False):
    """
    Args:
      cnn_process: The _CnnProcess to send captured images to.
      display: If true, it will enable a debugging display that shows the
               detected crops on-screen. """
    self.__cnn_process = cnn_process
    self.__display = display

    # Create the queues.
    self.__input_queue = Queue(maxsize=1)
    self.__output_queue = Queue()

    # Fork the capture process.
    self.__process = Process(target=_LandmarkProcess.run_forever,
                             args=(self,))
    self.__process.start()

  def release(self):
    """ Cleans up and terminates internal process. """
    self.__process.terminate()

  def __run_once(self):
    """ Reads and crops a single image. It will send it to the predictor
    process when finished. """
    # Get the next input from the queue.
    image, seq_num, timestamp = self.__input_queue.get()
    if image is None:
      # A None tuple means the end of a sequence. Propagate this through the
      # pipeline.
      self.__cnn_process.add_new_input(None, None, None, None, None, None, None)
      return
    if _is_stale(timestamp):
      # Image is stale. Indicate that it is invalid.
      self.__cnn_process.add_new_input(None, None, None, None, None, seq_num,
                                       timestamp)

    # Crop the image.
    left_eye, right_eye, face = self.__cropper.crop_image(image)
    if face is None:
      # We failed to get an image.
      logger.warning("Failed to get good detection for %d." % (seq_num))
      # Send along the sequence number.
      self.__cnn_process.add_new_input(None, None, None, None, None, seq_num,
                                       timestamp)
      return

    # Produce face mask.
    mask = self.__cropper.face_grid()
    # Estimate the head pose.
    head_pose = self.__cropper.estimate_pose()
    head_pose = head_pose[:, 0]

    if self.__display:
      # Show the debugging display.
      mask_sized = cv2.resize(mask, (224, 224))
      mask_sized = np.expand_dims(mask_sized, axis=2)
      mask_sized = np.tile(mask_sized, [1, 1, 3]) * 225.0
      mask_sized = mask_sized.astype(np.uint8)

      combined = np.concatenate((left_eye, right_eye, face, mask_sized), axis=1)
      cv2.imshow("Server Detections", combined)
      cv2.waitKey(1)

    # Send it along.
    self.__cnn_process.add_new_input(left_eye, right_eye, face, mask,
                                     head_pose, seq_num, timestamp)

  def run_forever(self):
    """ Reads and crops images indefinitely. """
    # Eye cropper to use for eye detection.
    self.__cropper = EyeCropper()

    while True:
      self.__run_once()

  def add_new_input(self, image, seq_num, timestamp):
    """ Adds a new input to be processed. Will block.
    Args:
      image: The image to process.
      seq_num: The sequence number of the image.
      timestamp: The timestamp of the image. """
    self.__input_queue.put((image, seq_num, timestamp))

class _CaptureProcess(object):
  """ Process that captures images from the camera, sends them over a queue. """

  def __init__(self, landmark_process, camera=-1):
    """
    Args:
      landmark_process: The landmark detection process to send images to.
      camera: The camera device to capture from. """
    self.__landmark_process = landmark_process
    self.__camera = camera

    # Fork the capture process.
    self.__process = Process(target=_CaptureProcess.run_forever,
                             args=(self,))
    self.__process.start()

  def __run_once(self):
    """ Runs a single iteration of the capture process. """
    ret, image = self.__capture.read()
    if not ret:
      raise RuntimeError("Could not read from camera.")

    # Send the image along.
    timestamp = time.time()
    self.__landmark_process.add_new_input(image, self.__sequence_num, timestamp)

    # Update the sequence number.
    self.__sequence_num += 1
    self.__sequence_num %= 255

  def run_forever(self):
    """ Runs the process indefinitely. """
    # Capture device to use.
    self.__capture = cv2.VideoCapture(self.__camera)
    # Sequence number for images.
    self.__sequence_num = 0

    while True:
      self.__run_once()

  def release(self):
    """ Cleans up and terminates internal process. """
    self.__process.terminate()
