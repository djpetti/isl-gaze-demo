import os

import keras.backend as K

import tensorflow as tf


class _Metric(object):
  """ Some metrics need extra state, so we implement them as a callable class,
  instead of a function. """

  @property
  def __name__(self):
    """ Keras expects this to be a function with a __name__ attribute, so we
    have to add one.
    Returns:
      The name of the class. """
    return self.__class__.__name__

  def __call__(self, y_true, y_pred):
    """ Implements the metric.
    Args:
      y_true: The "ground truth".
      y_pred: The predictions from the model. """
    raise NotImplementedError("__call__() must be implemented by subclass.")


class RefinedExamples(_Metric):
  """ Metric that writes refined images to the disk for qualitative analysis.
  (It only does so during the testing phase.) """

  def __init__(self, out_dir, max_to_write):
    """
    Args:
      out_dir: The output directory to write to.
      max_to_write: Maximum number of images to write to the disk each
                    iteration. These images will be replaced on the next one.
    """
    self.__max_to_write = max_to_write

    # Get the Keras graph.
    session = K.tensorflow_backend.get_session()
    self.__graph = session.graph

    # Pre-create file names for the resulting images.
    self.__image_names = []
    for i in range(0, self.__max_to_write):
      image_name = os.path.join(out_dir, "refined%d.jpg" % (i))
      self.__image_names.append(tf.constant(image_name))
    self.__image_names = tf.stack(self.__image_names)

  def __call__(self, y_true, y_pred):
    """ A special metric that writes examples of refined metrics to the disk. (It
    only does so during the testing phase.)
    Args:
      y_true: It expects this to be the input batch to the refiner.
      y_pred: It expects this to be the refined images.
    Returns:
      It always returns zero. The point of this metric is the side-effect it has
      of writing example images to the disk. """
    def do_disk_dump():
      """ Performs the actual disk dump of the images.
      Returns:
        Always returns 0. """
      return self.__write_batch(y_pred)

    # Exit immediately if we are training.
    default_return = K.constant(1.0)
    return K.in_test_phase(do_disk_dump, default_return)

  def __denormalize_image(self, image):
    """ Reverses the TF image normalization so that we can display the output of
    a refiner model for human evaluation.
    Args:
      image: The image to denormalize.
    Returns:
      The same image, denormalized. """
    image_min = tf.reduce_min(image)
    image_shifted = image - image_min

    image_max = tf.reduce_max(image_shifted)
    return image_shifted * 255.0 / image_max

  def __save_image(self, image, filename):
    """ JPEG-encodes an image, and saves it to a file.
    Args:
      image: The image to save.
      filename: The name of the file to save it to.
    Returns:
      The operation for writing to the file. """
    # Encode the image in JPEG form.
    encoded = tf.image.encode_jpeg(image)
    return tf.write_file(filename, encoded)

  def __write_batch(self, batch):
    """ Denormalizes and saves an entire batch of images.
    Args:
      batch: The batch to save. """
    def denormalize_and_write(image_pair):
      """ Denormalizes a single image and writes it to the disk.
      Args:
        image_pair: A tuple of the image and the filename. """
      image, filename = image_pair

      # Denormalize.
      image_denorm = self.__denormalize_image(image)
      # Convert to uint8.
      image_denorm = tf.cast(image_denorm, tf.uint8)
      # Write.
      write_op = self.__save_image(image_denorm, filename)

      # We need to force the write op to execute, so we link it to our constant
      # output.
      with self.__graph.control_dependencies([write_op]):
        ret = tf.constant(0.0)

      return ret

    # Only take the part of the batch that we need.
    save_part = batch[:self.__max_to_write]

    outputs = tf.map_fn(denormalize_and_write,
                        (save_part, self.__image_names),
                        dtype=tf.float32)
    return tf.reduce_sum(outputs)
