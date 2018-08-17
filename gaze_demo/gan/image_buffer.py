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

  def __get_buffer(self):
    """ Gets the updated buffer if we have one, or the buffer if we don't.
    Returns:
      The buffer. """
    if self.__buffer_update is None:
      return self.__buffer

    return self.__buffer_update

  def sample(self, size):
    """ Randomly samples images from the buffer.
    Args:
      size: The number of images to sample.
    Returns:
      An op producing a tensor of the sampled images. """
    # An easy way of implementing this is to randomly shuffle and then slice.
    my_buffer = self.__get_buffer()
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

    # Randomly select the images to keep.
    shuffled = tf.random_shuffle(my_buffer)
    num_to_keep = self.__get_buffer().shape[0] - new_images.shape[0]
    keep = shuffled[0:num_to_keep]

    # Create the new buffer contents.
    new_buffer = tf.concat([new_images, keep], 0)
    self.__buffer_update = self.__buffer.assign(new_buffer)

    return self.__buffer_update
