from keras.models import Model
import keras.layers as layers

from .. import network

import custom_layers


class GanNetwork(network.Network):
  """ Subclass of a standard network that (for now) ignores some of the inputs.
  """

  def _create_model(self):
    # Create the model using only the left eye input.
    model = Model(inputs=[self._left_eye_input], outputs=self._outputs)
    model.summary()

    return model

class RefinerNetwork(GanNetwork):
  """ Network that refines images from another person to make them look like
  images from me. """

  def _build_custom(self):
    # Convolutional layers.
    conv1 = layers.Conv2D(64, (3, 3), padding="SAME", activation="relu")
    block1 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    norm1 = layers.BatchNormalization()
    block2 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    norm2 = layers.BatchNormalization()
    block3 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    norm3 = layers.BatchNormalization()
    block4 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    norm4 = layers.BatchNormalization()
    block5 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    norm5 = layers.BatchNormalization()
    block6 = custom_layers.ResNetBlock(64, (3, 3), activation="relu")
    norm6 = layers.BatchNormalization()
    conv2 = layers.Conv2D(1, (1, 1))
    norm7 = layers.BatchNormalization()
    out = layers.Activation("tanh")

    # Apply the layers.
    r1 = conv1(self._left_eye_node)
    r2 = block1(r1)
    #r2 = norm1(r2)
    r3 = block2(r2)
    #r3 = norm2(r3)
    r4 = block3(r3)
    #r4 = norm3(r4)
    r5 = block4(r4)
    #r5 = norm4(r5)
    #r6 = block5(r5)
    #r6 = norm5(r6)
    #r7 = block6(r6)
    #r7 = norm6(r7)

    r8 = conv2(r5)
    #r8 = norm7(r8)
    #refined = out(r6)
    refined = r8

    return refined

class DescriminatorNetwork(GanNetwork):
  """ Network that attempts to distinguish between actual images of me and
  refined images of me. """

  def _build_custom(self):
    # Convolutional layers.
    conv1 = layers.Conv2D(96, (3, 3), strides=(2, 2), activation="relu")
    conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu")
    pool1 = layers.MaxPooling2D((3, 3), strides=(1, 1))
    norm1 = layers.BatchNormalization()

    conv3 = layers.Conv2D(32, (3, 3), activation="relu")
    conv4 = layers.Conv2D(32, (1, 1), activation="relu")
    conv5 = layers.Conv2D(2, (1, 1), activation="softmax")

    d1 = conv1(self._left_eye_node)
    d2 = conv2(d1)
    d3 = pool1(d2)
    #d3 = norm1(d3)
    d4 = conv3(d3)
    d5 = conv4(d4)
    logits = conv5(d5)

    return logits
