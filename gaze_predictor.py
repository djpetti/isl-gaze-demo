import cv2

from keras.models import load_model

import numpy as np

from face_tracking import landmark_detection as ld
from face_tracking import misc

from eye_cropper import EyeCropper
import image_tools
import metrics


class GazePredictor:
  """ Handles capturing eye images, and uses the model to predict gaze. """

  def __init__(self, model_file, capture_dev=-1, average_num=5):
    """
    Args:
      model_file: The saved model to load for predictions.
      capture_dev: The camera device to capture from.
      average_num: How many images to average in our predictions. Since we run
                   on the GPU, we can basically predict extra ones for free, but
                   it does take more time to capture and pre-process them. """
    self.__average_num = average_num

    # Eye cropper to use for eye detection.
    self.__cropper = EyeCropper()
    # Video capture instance to use for reading frames.
    self.__camera = cv2.VideoCapture(capture_dev)

    # Load the model we trained.
    custom = {"distance_metric": metrics.distance_metric,
              "accuracy_metric": metrics.accuracy_metric}
    self.__predictor = load_model(model_file, custom_objects=custom)

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

  def predict_gaze(self):
    """ Predicts the user's gaze based on current frames.
    Returns:
      The predicted gaze point, in normalized screen units, or (-1, -1) if it
      failed to predict the gaze. """
    # Get the input images.
    image_batch = []
    for _ in range(0, self.__average_num):
      eye_crop = self.__capture_eye()
      if eye_crop is None:
        # We failed to get an image.
        continue

      # Convert to black and white.
      eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
      # Normalize crop size.
      eye_crop = image_tools.reshape_image(eye_crop, (56, 26))
      image_batch.append(eye_crop)

    if not len(image_batch):
      # If we somehow managed to get no images, there's not a lot we can do.
      return (-1, -1)

    # Generate a prediction.
    image_batch = np.stack(image_batch)
    image_batch = np.expand_dims(image_batch, -1)

    # Do same pre-processing.
    image_batch = image_batch.astype(np.float32)
    image_batch -= 99
    image_batch /= np.std(image_batch)

    raw_preds = self.__predictor.predict(image_batch,
                                         batch_size=len(image_batch))

    # Average all the predictions to generate a final one.
    pred = np.mean(raw_preds, axis=0)
    return pred
