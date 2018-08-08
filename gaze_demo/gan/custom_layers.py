import keras.backend as K
import keras.layers as layers


class _SuperLayer(object):
  """ Allows the construction of a layer class comprised of other layers. """

  def __init__(self, layers):
    """
    Args:
      layers: A list of the sub-layers that this layer is comprised of. """
    self.__layers = layers

  def __call__(self, inputs):
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


class Residual(_SuperLayer):
  """ Implements a residual module as described by He et. al in
  "Deep residual learning for image recognition,"
	Computer Vision and Pattern Recognition, 2016. A layer wrapped with this class
	is evaluated as normal, but the original input is then added back to its
	output. """

  def __init__(self, layers, *args, **kwargs):
    """
    Args:
      layers: List of layers we want to wrap in a residual module. """
    super(Residual, self).__init__(layers, *args, **kwargs)

  def __check_shape_compatibility(self, inputs):
    """ Checks whether the input and output have compatible shapes.
    Args:
      The inputs to the module.
    Returns:
      The number of input filters, and the number of output filters. """
    input_shape = inputs._keras_shape
    output_shape = self.compute_output_shape(input_shape)

    # Assume order is (batch, h, w, filters) initially.
    input_size = input_shape[1:3]
    output_size = output_shape[1:3]
    input_filters = input_shape[3]
    output_filters = output_shape[3]

    if K.image_data_format() == "channels_first":
      # Order is actually (batch, filters, h, w)
      input_size = input_shape[2:]
      output_size = output_shape[2:]
      input_filters = input_shape[1]
      output_filters = output_shape[1]

    # Perform basic sanity checking.
    if (len(input_shape) != 4 or len(output_shape) != 4):
      raise ValueError("Residual() only works with 2D convolution.")
    if input_size != output_size:
      raise ValueError("Input and output image dims must be the same.")

    return input_filters, output_filters

  def __call__(self, inputs):
    # Add the wrapped layers.
    raw_outputs = super(Residual, self).__call__(inputs)

    add_back = inputs
    input_filters, output_filters = self.__check_shape_compatibility(inputs)
    if input_filters != output_filters:
      # We need an extra 1x1 convolution because the number of input and output
      # filters are not the same.
      add_back = layers.Conv2D(output_filters, (1, 1))

    # Perform the addition.
    return layers.Add()([raw_outputs, add_back])

class ResNetBlock(Residual):
  """ Performs a block of 2 convolution operations, wrapped in a residual
      module. """

  def __init__(self, filters, kernel_size, **kwargs):
    """ Creates a new block. Any additional arguments will be passed
    transparently to the underlying Conv2D layers. """
    # For now, it's easier not to mess with unpadded convolutions.
    if "padding" in kwargs:
      raise ValueError("Cannot specify padding for residual block.")
    kwargs["padding"] = "SAME"

    conv_args = [filters, kernel_size]

    # Initialize the sublayers.
    conv1 = layers.Conv2D(*conv_args, **kwargs)
    # We don't want activation for the last layer, since it will be added after
    # the addition operation.
    conv2_kwargs = kwargs.copy()
    conv2_kwargs["activation"] = None
    conv2 = layers.Conv2D(*conv_args, **conv2_kwargs)

    my_layers = [conv1, conv2]
    super(ResNetBlock, self).__init__(my_layers)
