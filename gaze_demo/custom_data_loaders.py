from pipeline import data_loader


class _GazeLoader(data_loader.DataLoader):
  """ A DataLoader subclass that implements the standard features for gaze
  recognition. """

  def _init_feature_set(self, prefix):
    features = data_loader.FeatureSet(prefix)

    features.add_feature("dots", tf.FixedLenFeature([2], tf.float32))
    features.add_feature("face_size", tf.FixedLenFeature([2], tf.float32))
    features.add_feature("leye_box", tf.FixedLenFeature([4], tf.float32))
    features.add_feature("reye_box", tf.FixedLenFeature([4], tf.float32))
    features.add_feature("grid_box", tf.FixedLenFeature([4], tf.float32))
    features.add_feature("image", tf.FixedLenFeature([1], tf.string))

    return features

class _GazeLoaderWithPose(_GazeLoader):
  """ A DataLoader subclass that implements the standard features for gaze
  recognition plus head pose. """

  def _init_feature_set(self, prefix):
    features = super(_GazeLoaderWithPose, self)._init_feature_set(prefix)

    features.add_feature("pose", tf.FixedLenFeature([3], tf.float32))
    return features

class TrainDataLoader(data_loader.TrainDataLoader, _GazeLoader)
