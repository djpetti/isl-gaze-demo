import keras.backend as K
import keras.layers as layers

import tensorflow as tf


class ImageBuffer(object):
  """ Implements a buffer for storing a history of refined images. """

  def __init__(self, size, image_shape):
    """
    Args:
      size: The number of images that the buffer holds.
      image_shape: The 3D shape of the images in the buffer. """
    # Create a variable to store the images.
    buffer_shape = (size,) + tuple(image_shape)
    self.__buffer = tf.Variable(tf.zeros(buffer_shape), name="buffer")
    # The update op for the buffer, when we have one.
    self.__buffer_update = None

    self.__buffer_size = size
    self.__image_shape = image_shape
    # In the case where the buffer is not yet full, this tracks the index of the
    # next empty space in the buffer.
    self.__fill_index = tf.Variable(0)

    # Get the Keras graph.
    session = K.tensorflow_backend.get_session()
    self.__graph = session.graph

  def __get_buffer(self):
    """ Gets the updated buffer if we have one, or the buffer if we don't.
    Returns:
      The buffer. """
    if self.__buffer_update is None:
      return self.__buffer

    return self.__buffer_update

  def is_saturated(self):
    """ Checks whether the buffer is full.
    Returns:
      A boolean Tensor indicating whether the buffer is full. """
    return tf.equal(self.__buffer_size, self.__fill_index)

  def supports_sample_size(self, size):
    """ Checks whether the buffer has enough data to support a sample of a given
    size.
    Args:
      size: The size to check for.
    Returns:
      A boolean tensor. True if the sample is supported, false otherwise. """
    return tf.greater(self.__fill_index, size)

  def sample(self, size):
    """ Randomly samples images from the buffer.
    Args:
      size: The number of images to sample.
    Returns:
      An op producing a tensor of the sampled images. """
    my_buffer = self.__get_buffer()
    # Don't use any parts of the buffer that aren't yet filled.
    my_buffer = my_buffer[:self.__fill_index]

    # An easy way of implementing this is to randomly shuffle and then slice.
    shuffled = tf.random_shuffle(my_buffer)
    return shuffled[0:size]

  def update(self, new_images):
    """ Replaces a set of random images from the buffer with images that were
    passed in.
    Args:
      new_images: The images to replace with.
    Returns:
      The buffer update operation. """
    my_buffer = self.__get_buffer()

    def replace():
      """ Replaces existing images with the new ones.
      Returns:
        The buffer update op. """
      # Randomly select the images to keep.
      shuffled = tf.random_shuffle(my_buffer)
      num_to_keep = self.__buffer_size - tf.shape(new_images)[0]
      keep = shuffled[0:num_to_keep]

      # Create the new buffer contents.
      new_buffer = tf.concat([new_images, keep], 0)

      update = self.__buffer.assign(new_buffer)

      return update

    def add():
      """ Adds new images to the buffer, without removing old ones. This should
      only be done when the buffer is not full.
      Returns:
        The buffer update op. """
      # Figure out how many images we need to fill the buffer.
      space_remaining = self.__buffer_size - self.__fill_index
      # How many new images we will use.
      num_new = tf.minimum(space_remaining, tf.shape(new_images)[0])
      to_add = new_images[:num_new]

      # Create the new buffer contents.
      existing_part = my_buffer[:self.__fill_index]
      # Pad the end if necessary.
      padding_size = self.__buffer_size - tf.shape(existing_part)[0] - \
                     tf.shape(to_add)[0]
      padding = tf.zeros_like(my_buffer)
      padding = padding[:padding_size]

      new_buffer = tf.concat([existing_part, to_add, padding], 0)

      # Update the fill index.
      fill_update = tf.assign_add(self.__fill_index, num_new)

      # Force the fill index update to actually run.
      with self.__graph.control_dependencies([fill_update]):
        update = self.__buffer.assign(new_buffer)

      return update

    # If the buffer is not full, we want to add to it until it fills up.
    # Otherwise, we replace existing images.
    self.__buffer_update = tf.cond(self.is_saturated(), replace, add)

    return self.__buffer_update
