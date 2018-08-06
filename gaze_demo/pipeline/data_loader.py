import os

import tensorflow as tf

import preprocess


class DataPoint(object):
  """ Structure encapsulating an image and associated metadata. """

  def __init__(self, features):
    """
    Args:
      features: The FeatureSet that we are using. """
    feature_names = features.get_feature_names()

    # Set all the properties.
    for name in feature_names:
      value = features.get_feature_by_name(name)
      setattr(self, name, value)

class FeatureSet(object):
  """ Defines a set of features that we want to load and process from the file.
  """

  def __init__(prefix, self):
    """
    Args:
      prefix: The prefix to use for all feature names. """
    self.__prefix = prefix

    # The feature specifications.
    self.__feature_specs = {}
    # The actual set of feature tensors.
    self.__features = {}

    # List of feature names.
    self.__feature_names = set()

  def add_feature(self, name, feature):
    """ Adds a feature to the set.
    Args:
      name: The name of the feature to add.
      feature: The TensorFlow feature. """
    self.__feature_names.add(name)

    # Add the prefix.
    full_name = "%s/%s" % (self.__prefix, name)
    self.__feature_specs[full_name] = feature

  def parse_from(self, batch):
    """ Parses all the features that were added.
    Args:
      batch: The batch input to parse features from. """
    self.__features = tf.parse_example(batch, features=self.__feature_specs)

  def get_features(self):
    """
    Returns:
      The full set of features. """
    return self.__features.copy()

  def get_feature_names(self):
    """
    Returns:
      A set of the names of all the features. """
    return self.__feature_names.copy()

  def get_feature_by_name(self, name):
    """ Gets a featue by its name. (Without the prefix.)
    Args:
      name: The name of the feature.
    Returns:
      The feature with that name. """
    # Add the prefix.
    full_name = "%s/%s" % (self.__prefix, name)
    return self.__features[full_name]


class DataLoader(object):
  """ Class that is responsible for loading and pre-processing data. """

  def __init__(self, records_file, batch_size, image_shape):
    """
    Args:
      records_file: The TFRecords file to read data from.
      batch_size: The size of batches to read.
      image_shape: The shape of images to load. """
    if not os.path.exists(records_file):
      # If we don't check this, TensorFlow gives us a really confusing and
      # hard-to-debug error later on.
      raise ValueError("File '%s' does not exist." % (records_file))

    self._image_shape = image_shape
    self._records_file = records_file
    self._batch_size = batch_size

    # Create a default preprocessing pipeline.
    self.__pipeline = preprocess.Pipeline()

  def __decode_and_preprocess(self):
    """ Target for map_fn that decodes and preprocesses individual images.
    Returns:
      A list of the preprocessed image nodes. """
    # Find the encoded image feature.
    jpeg = self._features.get_feature_by_name("image")

    # Decode the image.
    image = tf.image.decode_jpeg(jpeg[0])
    # Resize the image to a defined shape.
    image = tf.reshape(image, self._image_shape)

    # Create a data point object.
    data_point = DataPoint(features)
    # Use the decoded image instead of the encoded one.
    data_point.image = image

    # Pre-process the image.
    return self._build_preprocessing_stage(data_point)

  def __build_loader_stage(self):
    """ Builds the pipeline stages that actually loads data from the disk. """
    feature = {"%s/dots" % (prefix): tf.FixedLenFeature([2], tf.float32),
               "%s/face_size" % (prefix): tf.FixedLenFeature([2], tf.float32),
               "%s/leye_box" % (prefix): tf.FixedLenFeature([4], tf.float32),
               "%s/reye_box" % (prefix): tf.FixedLenFeature([4], tf.float32),
               "%s/grid_box" % (prefix): tf.FixedLenFeature([4], tf.float32),
               "%s/pose" % (prefix): tf.FixedLenFeature([3], tf.float32),
               "%s/image" % (prefix): tf.FixedLenFeature([1], tf.string)}

    # Create queue for filenames, which is a little silly since we only have one
    # file.
    filename_queue = tf.train.string_input_producer([self._records_file])

    # Define a reader and read the next record.
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    # Prepare random batches.
    batch = tf.train.shuffle_batch([serialized_examples],
                                   batch_size=self._batch_size,
                                   capacity=self._batch_size * 30,
                                   min_after_dequeue=self._batch_size * 15,
                                   num_threads=16)

    # Deserialize the example.
    self._features.parse_from(batch)

  def __associate_with_pipelines(self, out_nodes):
    """ Associates map_fn output nodes with their respective pipelines.
    Args:
      out_nodes: The output nodes from the map_fn call.
    Returns:
      A dictionary mapping pipelines to nodes. """
    pipelines = self.__pipeline.get_leaf_pipelines()

    mapping = {}
    for pipeline, node in zip(pipelines, out_nodes):
      mapping[pipeline] = node

    return mapping

  def _init_feature_set(self, prefix):
    """ Initializes the FeatureSet to use for this loader. This must be
    overriden by a subclass.
    Args:
      prefix: The prefix to use for feature names.
    Returns:
      The initialized FeatureSet. """
    raise NotImplementedError( \
        "_init_feature_set() must be implemented by subclass.")

  def _build_preprocessing_stage(self, data_point):
    """ Performs preprocessing on an image node.
    Args:
      data_point: The DataPoint object to use for preprocessing.
    Returns:
      The preprocessed image nodes. """
    # Convert the images to floats before preprocessing.
    data_point.image = tf.cast(data_point.image, tf.float32)

    # Build the entire pipeline.
    self.__pipeline.build(data_point)
    data_points = self.__pipeline.get_outputs()

    # Extract the image nodes.
    image_nodes = []
    for data_point in data_points:
      image_nodes.append(data_point.image)

    return image_nodes

  def _build_pipeline(self, prefix):
    """ Builds the entire pipeline for loading and preprocessing data.
    Args:
      prefix: The prefix that is used for the feature names. """
    # Initialize the feature set.
    self._init_feature_set(prefix)

    # Build the loader stage.
    self.__build_loader_stage(prefix)

    # Tensorflow expects us to tell it the shape of the output beforehand, so we
    # need to compute that.
    dtype = [tf.float32] * self.__pipeline.get_num_outputs()
    # Decode and pre-process in parallel.
    images = tf.map_fn(self.__decode_and_preprocess, dtype=dtype,
                       back_prop=False, parallel_iterations=16)

    # Create the batches.
    dots = self._features.get_feature_by_name("dots")
    self.__x = self.__associate_with_pipelines(images)
    self.__y = dots

  def get_data(self):
    """
    Returns:
      The loaded data, as a dict indexed by pipelines. """
    return self.__x

  def get_labels(self):
    """
    Returns:
      The node for the loaded labels. """
    return self.__y

  def get_pipeline(self):
    """ Gets the preprocessing pipeline object so that preprocessing stages can
    be added.
    Returns:
      The preprocessing pipeline. Add stages to this pipeline to control the
      preprocessing step. """
    return self.__pipeline

  def build(self):
    """ Builds the graph. This must be called before using the loader. """
    raise NotImplementedError("Must be implemented by subclass.")

class TrainDataLoader(DataLoader):
  """ DataLoader for training data. """

  def build(self):
    self._build_pipeline("train")

class TestDataLoader(DataLoader):
  """ DataLoader for testing data. """

  def build(self):
    self._build_pipeline("test")

class ValidDataLoader(DataLoader):
  """ DataLoader for validation data. """

  def build(self):
    self._build_pipeline("val")
