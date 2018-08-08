from keras.models import Model
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
