from keras.models import Model, load_model
import keras.applications as applications
import keras.backend as K
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers

import tensorflow as tf


class Network(object):
  """ Represents a network. """

  def __init__(self, input_shape, eye_shape=None, fine_tune=False,
               data_tensors=None, eye_preproc=None):
    """ Creates a new network.
    Args:
      input_shape: The input shape to the network.
      eye_shape: Specify the shape of the eye inputs, if it is different from
                 face input shape.
      fine_tune: Whether we are fine-tuning the model.
      data_tensors: If specified, the set of output tensors from the pipeline,
                    which will be used to build the model.
      eye_preproc: Optional custom layer to use for preprocessing the eye data.
                   If not present, it assumes that the input arrives
                   preprocessed."""
    self.__data_tensors = data_tensors
    self.__eye_preproc = eye_preproc
    self._fine_tune = fine_tune
    self._input_shape = input_shape

    self._eye_shape = self._input_shape
    if eye_shape is not None:
      self._eye_shape = eye_shape

  def _build_common(self):
    """ Build the network components that are common to all. """
    # L2 regularizer for weight decay.
    self._l2 = regularizers.l2(0.0005)

    leye = None
    reye = None
    face = None
    grid = None
    pose = None
    if self.__data_tensors:
      leye, reye, face, grid, pose = self.__data_tensors

    # Create inputs.
    self._left_eye_input = layers.Input(shape=self._eye_shape, tensor=leye,
                                        name="left_eye_input")
    self._right_eye_input = layers.Input(shape=self._eye_shape, tensor=reye,
                                         name="right_eye_input")
    self._face_input = layers.Input(shape=self._input_shape, tensor=face,
                                    name="face_input")
    self._grid_input = layers.Input(shape=(25, 25), tensor=grid,
                                    name="grid_input")
    self._pose_input = layers.Input(shape=(3,), tensor=pose, name="pose_input")

    # Add preprocessing layer.
    self._left_eye_node = self._left_eye_input
    self._right_eye_node = self._right_eye_input
    if self.__eye_preproc is not None:
      self._left_eye_node = self.__eye_preproc(self._left_eye_input)
      self._right_eye_node = self.__eye_preproc(self._right_eye_input)

  def _build_custom(self):
    """ Builds the custom part of the network. Override this in a subclass.
    Returns:
      The outputs that will be used in the model. """
    raise NotImplementedError("Must be implemented by subclass.")

  def build(self):
    """ Builds the network.
    Returns:
      The built model. """
    # Build the common parts.
    self._build_common()
    # Build the custom parts.
    outputs = self._build_custom()

    # Crea the model.
    model = Model(inputs=[self._left_eye_input, self._right_eye_input,
                          self._face_input, self._grid_input, self._pose_input],
                  outputs=outputs)
    model.summary()

    return model

class HeadPoseNetwork(Network):
  """ Simple network that incorporates the eyes and the estimated head pose. """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Shared eye layers.
    conv_e1 = layers.Convolution2D(50, (5, 5), strides=(1, 1),
                                   activation="relu")
    norm_e2 = layers.BatchNormalization()

    conv_e3 = layers.Convolution2D(100, (1, 1), activation="relu")
    norm_e4 = layers.BatchNormalization()
    conv_e5 = layers.Convolution2D(50, (1, 1), activation="relu")
    norm_e6 = layers.BatchNormalization()

    pool_e7 = layers.MaxPooling2D()

    conv_e8 = layers.Convolution2D(100, (5, 5), strides=(1, 1),
                                   activation="relu")
    norm_e9 = layers.BatchNormalization()

    pool_e10 = layers.MaxPooling2D()

    flat_e11 = layers.Flatten()

    # Left eye pathway.
    le1 = conv_e1(self._left_eye_node)
    le2 = norm_e2(le1)
    le3 = conv_e3(le2)
    le4 = norm_e4(le3)
    le5 = conv_e5(le4)
    le6 = norm_e6(le5)
    le7 = pool_e7(le6)
    le8 = conv_e8(le7)
    le9 = norm_e9(le8)
    le10 = pool_e10(le9)
    le11 = flat_e11(le10)

    # Right eye pathway.
    re1 = conv_e1(self._right_eye_node)
    re2 = norm_e2(re1)
    re3 = conv_e3(re2)
    re4 = norm_e4(re3)
    re5 = conv_e5(re4)
    re6 = norm_e6(re5)
    re7 = pool_e7(re6)
    re8 = conv_e8(re7)
    re9 = norm_e9(re8)
    re10 = pool_e10(re9)
    re11 = flat_e11(re10)

    # Fuse eye data.
    fused_eyes = layers.concatenate([le11, re11])
    fused_eyes = layers.Dense(100)(fused_eyes)

    # Head pose input.
    pose_values = layers.Dense(100, activation="relu")(self._pose_input)
    pose_values = layers.BatchNormalization()(pose_values)

    pose_values = layers.Dense(50, activation="relu")(pose_values)
    pose_values = layers.BatchNormalization()(pose_values)

    pose_values = layers.Dense(50, activation="relu")(pose_values)
    pose_values = layers.BatchNormalization()(pose_values)

    values = layers.concatenate([fused_eyes, pose_values])

    values = layers.Dense(256, activation="relu")(values)
    values = layers.BatchNormalization()(values)
    values = layers.Dense(128, activation="relu")(values)
    values = layers.BatchNormalization()(values)
    predictions = layers.Dense(2, activation="linear")(values)

    return predictions
