from keras.engine.topology import Layer
import keras.backend as K
import keras.layers as layers


class _SuperLayer(Layer):
  """ Allows the construction of a layer class comprised of other layers. """

  def build(self, input_shape, layers):
    """ Initializes the sublayers.
    Args:
      input_shape: The shape of the input to the layer.
      layers: The sublayers for this layer. """
    self.__layers = layers

    super(_SuperLayer, self).build(input_shape)

  def call(self, inputs):
    """ Actually adds to the graph for the layer.
    Returns:
      The layer output node. """
    # We'll do this by simply calling all the sublayers in sequence.
    for layer in self.__layers:
      inputs = layer(inputs)

    return inputs

  def compute_output_shape(self, input_shape):
    """ Computes the output shape of this layer given the input shape.
    Args:
      input_shape: The shape of the input.
    Returns:
      The shape of the corresponding output. """
    # We'll just do this by calling the same method on our sublayers in
    # sequence.
    for layer in self.__layers:
      input_shape = layer.compute_output_shape(input_shape)

    return input_shape

  def get_weights(self):
    """ Gets the weights for this layer.
    Returns:
      A list of all the weights in the sublayer. """
    all_weights = []
    for layer in self.__layers:
      all_weights.extend(layer.get_weights())

    return all_weights

  def get_sublayers(self):
    """ Gets the list of sublayers.
    Returns:
      The list of sublayers. """
    return self.__layers


class Residual(Layer):
  """ Implements a residual module as described by He et. al in
  "Deep residual learning for image recognition,"
	Computer Vision and Pattern Recognition, 2016. A layer wrapped with this class
	is evaluated as normal, but the original input is then added back to its
	output. """

  def __init__(self, wrapped_layer, *args, **kwargs):
    """
    Args:
      wrapped_lay: ng, S. Ren, and J. Sun.  Deep resid-
      ual learning for image recognition.The layer we are wrapping in the residual module. """
    super(Residual, self).__init__(*args, **kwargs)

    self.__layer = wrapped_layer

  def build(self, input_shape):
    """ Initializes the residual module.
    Args:
      input_shape: The shape of the input to the module. """
    super(Residual, self).build(input_shape)

    # The internal layer has to be built as well for this to work.
    self.__layer.build(input_shape)

    # Get the number of output filters for this layer.
    _, _, _, out_filters = self.__layer.compute_output_shape(input_shape)
    _, _, _, in_filters = input_shape

    self.__dim_mismatch = False
    if out_filters != in_filters:
      # We're going to need to do a 1x1 convolution to make them compatible.
      self.__dim_mismatch = True
      self.__conv = layers.Convolution2D(out_filters, (1, 1))

    # Post-addition activation layer.
    self.__activation = layers.Activation("relu")

  def call(self, inputs):
    """ Actually adds to the graph for the layer.
    Returns:
      The layer output node. """
    add = None
    if not self.__dim_mismatch:
      # We can do a straight passthrough here.
      add = inputs
    else:
      # We're going to need to change the shape.
      add = self.__conv(inputs)

    layer_outputs = self.__layer(inputs)
    return self.__activation(layer_outputs + add)

  def compute_output_shape(self, input_shape):
    """ Computes the output shape of this layer given the input shape.
    Args:
      input_shape: The shape of the input.
    Returns:
      The shape of the corresponding output. """
    # In this case, our output shape will be the same as that of our wrapped
    # layer.
    return self.__layer.compute_output_shape(input_shape)

  def get_weights(self):
    """ Gets the weights for this layer.
    Returns:
      The weights for the wrapped layer, with the weights for the pass-through
      convolution present, if active. """
    if not self.__dim_mismatch:
      # No pass-through convolution.
      return self.__layer.get_weights()
    else:
      return self.__layer.get_weights().extend(self.__conv.get_weights())

  def get_sublayers(self):
    """
    Returns:
      The list of sublayers that make up this layer. """
    if not self.__dim_mismatch:
      # No pass-through convolution.
      return [self.__layer]
    else:
      return [self.__layer, self.__conv]


class _ResNetConvBlock(_SuperLayer):
  """ Performs a block of convolution operations. This is the main part of a
  ResNet block. """

  def __init__(self, filters, kernel_size, **kwargs):
    """ Creates a new block. Any additional arguments will be passed
    transparently to the underlying Conv2D layers. """
    super(_ResNetConvBlock, self).__init__()

    self.__conv_args = [filters, kernel_size]
    self.__conv_kwargs = kwargs

  def build(self, input_shape):
    """ Defines weights and initializes sub-layers.
    Args:
      input_shape: The shape of the layer inputs. """
    # Initialize the sublayers.
    conv1 = layers.Conv2D(*self.__conv_args, **self.__conv_kwargs)
    # We don't want activation for the last layer, since it will be added after
    # the addition operation.
    conv2_kwargs = self.__conv_kwargs[:]
    conv2_kwargs["activation"] = None
    conv2 = layers.Conv2D(*self.__conv_args, **conv2_kwargs)

    my_layers = [conv1, conv2]
    super(_ResNetConvBlock, self).build(input_shape, my_layers)

class ResNetBlock(_SuperLayer):
  """ Wraps a convolution block in a residual layer to form a ResNet residual
  module. """

  def __init__(self, *args, **kwargs):
    """ Creates a new block. Any additional arguments will be passed
    transparently to the underlying Conv2D layers. """
    super(ResNetBlock, self).__init__()

    self.__conv_args = args
    self.__conv_kwargs = kwargs

  def build(self, input_shape):
    """ Defines weights and initializes sub-layers.
    Args:
      input_shape: The shape of the layer inputs. """
    layer = Residual(_ResNetConvBlock(*self.__conv_args, **self.__conv_kwargs))

    super(ResNetBlock, self).build(input_shape, [layer])


