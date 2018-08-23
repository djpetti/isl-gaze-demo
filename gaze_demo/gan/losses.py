import keras.backend as K
import keras.metrics as metrics

import tensorflow as tf

import utils


class _Loss(object):
  """ Some losses need extra state, so we implement them as a callable class,
  instead of a function. """

  @property
  def __name__(self):
    """ Keras expects this to be a function with a __name__ attribute, so we
    have to add one.
    Returns:
      The name of the class. """
    return self.__class__.__name__

  def __call__(self, y_true, y_pred):
    """ Implements the loss.
    Args:
      y_true: The "ground truth".
      y_pred: The predictions from the model. """
    raise NotImplementedError("__call__() must be implemented by subclass.")


class RealismLoss(_Loss):
  """ This is a custom loss function that implements the GAN realism loss. """

  def __init__(self, descrim_model):
    """
    Args:
      descrim_model: The model to use for determining whether an image is real
                     or synthetic. """
    self.__descrim_model = descrim_model

  def __call__(self, y_true, y_pred):
    """ Implements the realism loss.
    Args:
      y_true: It expects this to be the original (unrefined) images.
      y_pred: It expects this to be the refined images.
    Returns:
      A value for the realism loss. """
    # Get the descriminator classification.
    descrim_output = self.__descrim_model([y_pred])

    # Calculate cross-entropy loss. In this case, the "right" answer is all of
    # them being real.
    labels = utils.make_real_labels(descrim_output.shape)
    loss = K.binary_crossentropy(labels, descrim_output)

    # Sum over all positions.
    loss = K.sum(loss, axis=3)
    loss = K.sum(loss, axis=2)
    loss = K.sum(loss, axis=1)

    loss = K.print_tensor(loss, message="Real loss: ")

    return loss

class RegularizationLoss(_Loss):
  """ This is a custom loss function that implements the GAN regularization
  loss. """

  def __init__(self, scale):
    """
    Args:
      scale: The scale factor for the regularization. """
    self.__reg_scale = scale

  def __l1_norm(self, y_true, y_pred):
    """ Takes the L1 norm.
    Args:
      y_true: It expects this to be the original (unrefined) images.
      y_pred: It expects this to be the refined images.
    Returns:
      The L1 norm . """
    norm = K.abs(y_pred - y_true)
    norm = K.sum(norm, axis=3)
    norm = K.sum(norm, axis=2)
    norm = K.sum(norm, axis=1)

    return norm

  def __standardize(self, inputs):
    """ Standardizes the input by subtracting the mean and dividing by the
    standard deviation.
    Args:
      inputs: The inputs to standardize.
    Returns:
      The standardized inputs. """
    normalized, _, _ = K.normalize_batch_in_training(inputs, 1.0, 0.0,
                                                     [0, 1, 2, 3])
    return normalized

  def __call__(self, y_true, y_pred):
    """ Implements the regularization loss.
    Args:
      y_true: It expects this to be the original (unrefined) images.
      y_pred: It expects this to be the refined images.
    Returns:
      A value for the regularization loss. """
    # Standardize.
    #y_true = self.__standardize(y_true)
    #y_pred = self.__standardize(y_pred)

    # Take the L1 norm.
    norm = self.__l1_norm(y_true, y_pred)

    # Scale the loss.
    loss = self.__reg_scale * norm
    loss = K.print_tensor(loss, message="Reg loss: ")
    return loss

class CombinedLoss(_Loss):
  """ This is a custom loss function for the GAN that combines both realism and
  regularization loss. """

  def __init__(self, descrim_model, scale):
    """
    Args:
      descrim_model: The model to use for determining whether an image is real
                     or synthetic.
      scale: The scale factor for the regularization. """
    self.__realism_loss = RealismLoss(descrim_model)
    self.__regularization_loss = RegularizationLoss(scale)

  def __call__(self, y_true, y_pred):
    """ Implements the GAN loss.
    Args:
      y_true: It expects this to be the original (unrefined) images.
      y_pred: It expects this to be the refined images.
    Returns:
      A value for the overall loss. """
    realism = self.__realism_loss(y_true, y_pred)
    regularization = self.__regularization_loss(y_true, y_pred)

    return realism + regularization
