from keras.models import Model
import keras.backend as K
import keras.initializers as initializers
import keras.layers as layers
import keras.regularizers as regularizers

import numpy as np

import config


def _create_bitmask_image(x1, y1, x2, y2):
  """ Creates the bitmask image from the bbox points.
  Args:
    x1, y1: The x and y coordinates of the first point, in frame fractions.
    x2, y2: The x and y coordinates of the second point, in frame fractions.
  Returns:
    The generated bitmask image. """
  # Scale to mask size.
  x1 *= 25
  y1 *= 25
  x2 *= 25
  y2 *= 25

  x1 = int(x1)
  y1 = int(y1)
  x2 = int(x2)
  y2 = int(y2)

  # Create the interior image.
  width = x2 - x1
  height = y2 - y1
  face_box = np.ones((height, width))

  # Create the background.
  frame = np.zeros((25, 25))
  # Superimpose it correctly.
  frame[y1:y2, x1:x2] = face_box

  return frame


def convert_labels(labels, gaze_only=False):
  """ Convert the raw labels from the dataset into matrices that can be fed into
  the loss function.
  Args:
    labels: The labels to convert.
    gaze_only: If true, it will only extract the gaze point, and not the head
    pose and position.
  Returns:
    The converted label gaze points, poses, and face masks. """
  num_labels = []
  poses = []
  face_masks = []
  for label in labels:
    if not gaze_only:
      coords, pitch, yaw, roll, x1, y1, x2, y2 = label.split("_")[:8]
    else:
      coords = label.split("_")[0]

    x_pos, y_pos = coords.split("x")
    x_pos = float(x_pos)
    y_pos = float(y_pos)

    # Scale to screen size. We divide by the smaller dimension only so that we
    # end up with a 1-to-1 mapping between loss and pixel error.
    small_dim = min(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    x_pos /= small_dim
    y_pos /= small_dim

    num_labels.append([x_pos, y_pos])

    if not gaze_only:
      # Convert poses.
      pitch = float(pitch)
      yaw = float(yaw)
      roll = float(roll)

      poses.append([pitch, yaw, roll])

      # Convert bitmask.
      x1 = float(x1)
      y1 = float(y1)
      x2 = float(x2)
      y2 = float(y2)

      face_mask = _create_bitmask_image(x1, y1, x2, y2)
      face_masks.append(face_mask)

  stack = np.stack(num_labels, axis=0)
  if not gaze_only:
    pose_stack = np.stack(poses, axis=0)
    face_stack = np.stack(face_masks, axis=0)
  else:
    pose_stack = np.zeros((len(num_labels), 3))
    face_stack = np.zeros((len(num_labels), 25, 25))

  return (stack, pose_stack, face_stack)


def stddev_layer(layer_in):
  """ Divides the input by its standard deviation.
  Args:
    layer_in: The input tensor.
  Returns:
    The input divided by its standard deviation. """
  return layer_in / K.std(layer_in)

def bw_layer(layer_in):
  """ Converts the input to black-and-white.
  Args:
    layer_in: The input tensor.
  Returns:
    A black-and-white version of the input. """
  bw = layer_in[:, :, :, 0] * 0.288 + \
       layer_in[:, :, :, 1] * 0.587 + \
       layer_in[:, :, :, 2] * 0.114
  return K.expand_dims(bw, 3)

def nin_layer(num_filters, filter_size, top_layer, padding="valid", l2=0):
  """ Creates a network-in-network layer.
  Args:
    num_filters: The number of output filters.
    filter_size: The size of the filters.
    top_layer: The layer to build off of.
    padding: The type of padding to use.
    l2: Amount of l2 regularization.
  Returns:
    The network-in-network layer output node. """
  l2_reg = regularizers.l2

  conv = layers.Conv2D(num_filters, filter_size, kernel_regularizer=l2_reg(l2),
                       padding=padding,
                       kernel_initializer=deep_xavier())(top_layer)
  # Share across all parameters in a filter.
  act = layers.advanced_activations.PReLU(shared_axes=[1, 2])(conv)
  norm = layers.BatchNormalization()(act)

  conv2 = layers.Conv2D(num_filters * 2, (1, 1), kernel_regularizer=l2_reg(l2),
                        kernel_initializer=deep_xavier())(norm)
  act2 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(conv2)
  norm2 = layers.BatchNormalization()(act2)

  conv3 = layers.Conv2D(num_filters, (1, 1), kernel_regularizer=l2_reg(l2),
                        kernel_initializer=deep_xavier())(norm2)
  act3 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(conv3)
  norm3 = layers.BatchNormalization()(act3)

  return norm3

def resid_module(num_filters, filter_size, top_layer, padding="valid", l2=0):
  """ Creates a residual module.
  Args:
    num_filters: The number of output filters.
    filter_size: The size of the filters.
    top_layer: The layer to build off of.
    padding: The type of padding to use.
    l2: Amount of l2 regularization.
  Returns:
    The residual module output node. """
  # Core conv layer.
  nin = nin_layer(num_filters, filter_size, top_layer, padding=padding, l2=l2)

  # Add a 1x1 conv if we need the sizes to sync up.
  resid = top_layer
  _, height, width, channels = K.int_shape(top_layer)
  if (channels is None or channels != num_filters):
    if not channels:
      print "WARNING: Unable to detect number of channels for top layer."

    resid = layers.Conv2D(num_filters, (1, 1))(top_layer)

  if padding == "valid":
    # In this case, we also need to crop to get the sizes to match.
    filter_w, filter_h = filter_size
    new_w = width - filter_w + 1
    new_h = height - filter_h + 1

    symmetric_w = (width - new_w) / 2
    symmetric_h = (height - new_h) / 2

    resid = layers.Cropping2D((symmetric_h, symmetric_w))(resid)

  return layers.add([nin, resid])


def deep_xavier(seed=None):
  """ This is a special variation of Glorot (normal) initialization that is
  specifically designed to aid convergence in deep ReLU-activated networks.

  It is essentially the same as standard Glorot initialization, except the
  variance is scaled by a factor of 2. In other words,
  `stddev = sqrt(2 / fan_in)`, where `fan_in` is the number of input units in
  the weight tensor.

  # Arguments
    seed: A Python integer. Used to seed the random generator.

  # Returns:
    An initializer.

  # References
    He et. al, cs.CV 2015
    https://arxiv.org/pdf/1502.01852.pdf
  """
  return initializers.VarianceScaling(scale=2.,
                                      mode='fan_in',
                                      distribution='normal',
                                      seed=seed)



def build_network(input_shape, l2=0):
  """
  Builds the network from scratch.
  Args:
    input_shape: The shape of the network inputs.
    l2: The l2 regularization to use.
  Returns:
    The built network, not yet compiled. """
  inputs = layers.Input(shape=input_shape, name="main_input")

  floats = K.cast(inputs, "float32")
  noisy = layers.GaussianNoise(0)(floats)

  noisy = layers.Lambda(bw_layer)(noisy)
  noisy = layers.Lambda(stddev_layer)(noisy)

  mod1 = resid_module(50, (5, 5), noisy, padding="same", l2=l2)

  values = layers.Conv2D(50, (3, 3), strides=(2, 2), padding="same")(mod1)

  mod2 = resid_module(100, (5, 5), values, padding="same", l2=l2)
  mod5 = resid_module(150, (5, 5), mod2, padding="same", l2=l2)
  mod3 = resid_module(200, (5, 5), mod5)

  pool2 = layers.Conv2D(200, (3, 3), strides=(2, 2), padding="same")(mod3)

  mod4 = resid_module(200, (3, 3), pool2, padding="same", l2=l2)
  mod6 = resid_module(200, (3, 3), mod4, padding="same", l2=l2)

  # Squeeze the number of filters so the FC part isn't so huge.
  values = layers.Conv2D(75, (1, 1),
                         kernel_initializer=deep_xavier())(mod6)
  values = layers.advanced_activations.PReLU(shared_axes=[1, 2])(values)
  values = layers.BatchNormalization()(values)

  values = layers.Flatten()(values)

  values = layers.Dense(150, kernel_initializer=deep_xavier())(values)
  values = layers.advanced_activations.PReLU()(values)
  values = layers.BatchNormalization()(values)

  # Head pose input.
  pose_input = layers.Input(shape=(3,), name="pose_input")
  pose_values = layers.Dense(100, activation="relu",
                             kernel_initializer=deep_xavier())(pose_input)
  pose_values = layers.BatchNormalization()(pose_values)

  pose_values = layers.Dense(75, activation="relu",
                             kernel_initializer=deep_xavier())(pose_values)
  pose_values = layers.BatchNormalization()(pose_values)

  pose_values = layers.Dense(75, activation="relu",
                             kernel_initializer=deep_xavier())(pose_values)
  pose_values = layers.BatchNormalization()(pose_values)

  # Face mask input.
  mask_input = layers.Input(shape=(25, 25), name="mask_input")

  # We have to flatten the masks before we can use them in the FF layers.
  mask_values = layers.Flatten()(mask_input)

  mask_values = layers.Dense(100, activation="relu",
                             kernel_initializer=deep_xavier())(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  mask_values = layers.Dense(75, activation="relu",
                             kernel_initializer=deep_xavier())(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  mask_values = layers.Dense(75, activation="relu",
                             kernel_initializer=deep_xavier())(mask_values)
  mask_values = layers.BatchNormalization()(mask_values)

  values = layers.concatenate([values, pose_values, mask_values])

  values = layers.Dense(300, activation="relu",
                        kernel_initializer=deep_xavier())(values)
  values = layers.BatchNormalization()(values)
  values = layers.Dropout(0.5)(values)

  values = layers.Dense(200, activation="relu",
                        kernel_initializer=deep_xavier())(values)
  values = layers.Dropout(0.5)(values)
  values = layers.BatchNormalization()(values)
  predictions = layers.Dense(2, activation="linear")(values)

  model = Model(inputs=[inputs, pose_input, mask_input], outputs=predictions)

  return model


def train_once(model, data, batch_size, use_aux=True):
  """ Run a single training iteration.
  Args:
    model: The model to train.
    data: The data loader to get data from.
    batch_size: Batch size to use for training.
    use_aux: Use auxiliary data, i.e. head pose and pos.
  Returns:
    The training loss. """
  # Get a new chunk of training data.
  training_data, training_labels = data.get_train_set()
  # Convert to a usable form.
  training_labels, pose_data, mask_data = \
      convert_labels(training_labels, gaze_only=not use_aux)

  # Train the model.
  history = model.fit([training_data, pose_data, mask_data],
                      training_labels,
                      epochs=1,
                      batch_size=batch_size)

  return history.history["loss"]

def test_once(model, data, batch_size, use_aux=True):
  """ Run a single testing iteration.
  Args:
    model: The model to train.
    data: The data loader to get data from.
    batch_size: Batch size to use for testing.
    use_aux: Use auxiliary data, i.e. head pose and pos.
  Returns:
    The testing loss and accuracy. """
  testing_data, testing_labels = data.get_test_set()
  testing_labels, pose_data, mask_data = \
      convert_labels(testing_labels, gaze_only=not use_aux)

  loss, accuracy = model.evaluate([testing_data, pose_data, mask_data],
                                   testing_labels,
                                   batch_size=batch_size)

  return loss, accuracy
