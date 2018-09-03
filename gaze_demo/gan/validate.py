import logging

import keras.backend as K
import keras.optimizers as optimizers

import tensorflow as tf

from .. import config, metrics

import model
import pipelines

# The shape that we expect for raw images loaded from the disk.
RAW_IMAGE_SHAPE = (400, 400, 3)
# The shape of the images that are inputs to the networks.
EST_INPUT_SHAPE = (32, 32, 1)
REF_INPUT_SHAPE = (44, 44, 1)


# Configure GPU VRAM usage.
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
g_session = tf.Session(config=tf_config)
K.tensorflow_backend.set_session(g_session)


logger = logging.getLogger(__name__)


class GanValidator(object):
  """ Class that handles validating the trained GAN. """

  def __init__(self, parser):
    """
    Args:
      parser: The CLI argument parser. """
    self.__parser = parser
    self.__args = self.__parser.parse_args()

  def __do_validation(self):
    """ Performs the actual validation, once everything is initialized. """
    # Create a coordinator and run queues.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=g_session)

    all_accuracy = []

    for i in range(0, self.__args.valid_iters):
      loss, acc = self.__estimator_model.evaluate(steps=self.__args.iter_steps)

      logger.info("Loss: %f, Accuracy: %f" % (loss, acc))
      all_accuracy.append(acc)

    # Compute the average accuracy.
    average = float(sum(all_accuracy)) / len(all_accuracy)
    logger.info("Average accuracy: %f" % average)

    coord.request_stop()
    coord.join()

  def __build_input_pipelines(self, output_shape):
    """ Builds the input pipelines.
    Args:
      output_shape: The shape of the images that we want to get out of the
                    pipeline. """
    builder = pipelines.PipelineBuilder(RAW_IMAGE_SHAPE, output_shape[:2],
                                        self.__args.batch_size)
    isl_input_tensors = builder.build_pipeline(self.__args.train_set,
                                               self.__args.test_set,
                                               has_pose=True)

    # Separate into inputs and labels.
    self.__data_tensors = isl_input_tensors[:5]
    self.__label_tensor = isl_input_tensors[5]

  def __build_estimator(self, input_tensors):
    """ Builds the estimator model, and loads the saved weights.
    Args:
      input_tensors: The tensors to use for the model's input. """
    estimator = config.NET_ARCH(EST_INPUT_SHAPE,
                                data_tensors=input_tensors)
    self.__estimator_model = estimator.build()

    # Load the saved weights.
    self.__estimator_model.load_weights(self.__args.est_model)

    # Since we don't actually train, we're going to force it to constantly be in
    # training phase. That way, it will use the larger training dataset for
    # evaluation. This doesn't matter because the models have never seen any of
    # the images in this dataset.
    K.set_learning_phase(1)

    # Compile the model. The optimizer doesn't really matter, since we're not
    # training anyway.
    opt = optimizers.SGD(lr=1.0)
    self.__estimator_model.compile(optimizer=opt, loss=metrics.distance_metric,
                                   metrics=[metrics.accuracy_metric],
                                   target_tensors=[self.__label_tensor])

  def __build_refiner(self):
    """ Builds the refiner model, and loads the saved weights. Note that it does
    not compile it, because it is intended to be used as a layer. """
    # Build the refiner.
    refiner = model.RefinerNetwork(REF_INPUT_SHAPE)
    self.__refiner_model = refiner.build()

    # Load the saved weights.
    self.__refiner_model.load_weights(self.__args.ref_model)

  def validate_baseline(self):
    """ Performs a "baseline" validation, operating on images directly without
    the GAN refiner in the loop. """
    logger.info("Performing baseline validation.")

    # Build input pipelines and estimator.
    self.__build_input_pipelines(EST_INPUT_SHAPE)
    self.__build_estimator(self.__data_tensors)

    # Perform the validation.
    self.__do_validation()

  def validate_gan(self):
    """ Performs validation with the GAN refiner in the loop. """
    logger.info("Performing validation with GAN refiner.")

    # Build input pipelines and refiner.
    self.__build_input_pipelines(REF_INPUT_SHAPE)
    self.__build_refiner()

    # Use the refiner to "preprocess" data.
    refined = self.__refiner_model(self.__data_tensors[:1])

    # TODO (danielp): Make both networks use the same size input.
    refined = tf.image.resize_images(refined, EST_INPUT_SHAPE[:2],
                                     align_corners=True)

    # Build the estimator off of the refined data.
    input_tensors = [refined] + self.__data_tensors[1:]
    self.__build_estimator(input_tensors)

    # Perform the validation.
    self.__do_validation()

  def validate(self):
    """ Either performs baseline or GAN validation depending on what the user
    selected. """
    if self.__args.baseline:
      self.validate_baseline()
    else:
      self.validate_gan()
