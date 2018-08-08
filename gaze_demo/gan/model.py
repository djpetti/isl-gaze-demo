from keras.models import Model
import keras.backend as K
import keras.layers as layers

from .. import network

import custom_layers


class GanNetwork(network.Network):
  """ Subclass of a standard network that (for now) ignores some of the inputs.
  """

  def build(self):
    # Build the common parts.
    self._build_common()
    # Build the custom parts.
    outputs = self._build_custom()

    # Create the model.
    model = Model(inputs=[self._left_eye_input], outputs=outputs)
    model.summary()

    return model

class RefinerNetwork(GanNetwork):
  """ Network that refines images from another person to make them look like
  images from me. """

  def _build_custom(self):
    # Convolutional layers.
    conv1 = layers.Conv2D(64, (3, 3), padding="SAME", activation="relu")
    block1 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    block2 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    block3 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    block4 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    conv2 = layers.Conv2D(1, (1, 1))
    out = layers.Activation("tanh")
    relu = layers.Activation("relu")

    # Apply the layers.
    r1 = conv1(self._left_eye_node)
    r2 = block1(r1)
    r3 = block2(r2)
    r4 = block3(r3)
    r5 = block4(r4)

    r6 = conv2(r5)
    #refined = out(r6)
    refined = r6

    return refined

class DescriminatorNetwork(GanNetwork):
  """ Network that attempts to distinguish between actual images of me and
  refined images of me. """

  def _build_custom(self):
    # Convolutional layers.
    conv1 = layers.Conv2D(96, (3, 3), strides=(2, 2), activation="relu")
    conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu")
    pool1 = layers.MaxPooling2D((3, 3), strides=(1, 1))

    conv3 = layers.Conv2D(32, (3, 3), activation="relu")
    conv4 = layers.Conv2D(32, (1, 1), activation="relu")
    conv5 = layers.Conv2D(2, (1, 1), activation="softmax")

    d1 = conv1(self._left_eye_node)
    d2 = conv2(d1)
    d3 = pool1(d2)
    d4 = conv3(d3)
    d5 = conv4(d4)
    logits = conv5(d5)

    return logits


class AdversarialLoss(object):
  """ This is a custom loss function for the GAN. We implement it as a callable
  class instead of a normal function because it needs some extra state. """

  def __init__(self, descrim_model, scale):
    """
    Args:
      descrim_model: The model to use for determining whether an image is real
                     or synthetic.
      scale: The scale factor for the regularization. """
    self.__descrim_model = descrim_model
    self.__reg_scale = scale

  @property
  def __name__(self):
    """ Keras expects this to be a function with a __name__ attribute, so we
    have to add one.
    Returns:
      The name of the class. """
    return self.__class__.__name__

  def __call__(self, y_true, y_pred):
    """ Implements the GAN loss.
    Args:
      y_true: It expects this to be the original (unrefined) images.
      y_pred: It expects this to be the refined images.
    Returns:
      A value for the overall loss. """
    realism = self.__realism_loss(y_pred)
    regularization = self.__regularization_loss(y_true, y_pred)

    return regularization
    return realism + regularization

  def __realism_loss(self, refined):
    """ Calculates the value of the realism term.
    Args:
      refined: The refined images.
    Returns:
      The value of the realism loss for these images. """
    # Get the descriminator classification.
    descrim_output = self.__descrim_model([refined])

    # Calculate cross-entropy loss. In this case, the "right" answer is all of
    # them being real.
    labels = K.zeros_like(descrim_output)
    loss = K.binary_crossentropy(labels, descrim_output)

    # Sum over all positions.
    loss = K.sum(loss, axis=3)
    loss = K.sum(loss, axis=2)
    loss = K.sum(loss, axis=1)

    return loss

  def __regularization_loss(self, raw, refined):
    """ Calculates the value of the regularization term.
    Args:
      raw: The raw inputs to the refiner network.
      refined: The outputs from the refiner network.
    Returns:
      The value of the regularization loss for these images. """
    # Take the L1 norm.
    norm = K.abs(refined - raw)
    norm = K.sum(norm, axis=3)
    norm = K.sum(norm, axis=2)
    norm = K.sum(norm, axis=1)

    # Scale the loss.
    return self.__reg_scale * norm
