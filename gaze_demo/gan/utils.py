import tensorflow as tf


def _make_slices(shape):
  """ Makes the 3D zeros and ones slices that we use to construct labels.
  Args:
    shape: The shape of the slices. (4th dim will be ignored.)
  Returns:
    The zeros slice and ones slice. """
  if len(shape) != 4:
    raise ValueError("Labels shape must be 4D.")

  zeros_slice = tf.zeros(shape)[:, :, :, 0]
  ones_slice = tf.ones(shape)[:, :, :, 0]

  return (zeros_slice, ones_slice)

def make_real_labels(shape):
  """ Creates a label tensor indicating that all samples are real.
  Args:
    shape: The shape of the label tensor.
  Returns:
    Real label tensor. """
  zeros_slice, ones_slice = _make_slices(shape)
  return tf.stack([zeros_slice, ones_slice], 3)

def make_fake_labels(shape):
  """ Creates a label tensor indicating that all samples are fake.
  Args:
    shape: The shape of the label tensor.
  Returns:
    Fake label tensor. """
  zeros_slice, ones_slice = _make_slices(shape)
  return tf.stack([ones_slice, zeros_slice], 3)
