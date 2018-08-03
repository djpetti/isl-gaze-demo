from keras import backend as K

import config


def distance_metric(y_true, y_pred):
  """ Calculates the euclidean distance between the two labels and the
  predictions.
  Args:
    y_true: The true labels.
    y_pred: The predictions.
  Returns:
    The element-wise euclidean distance between the labels and the predictions.
  """
  diff = y_true - y_pred
  sqr = K.square(diff)
  total = K.sum(sqr, axis=1)
  return K.sqrt(total)

def accuracy_metric(y_true, y_pred):
  """ Calculates the accuracy, converting back to pixel values.
  Args:
    y_true: The true labels.
    y_pred: The predictions.
  Returns:
    The element-wise euclidean distance between the labels and the predictions.
  """
  # Scale to actual pixel values.
  screen_size = K.constant([config.SCREEN_WIDTH, config.SCREEN_HEIGHT])
  y_true *= screen_size
  y_pred *= screen_size

  return distance_metric(y_true, y_pred)

