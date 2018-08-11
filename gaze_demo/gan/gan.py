import logging
import os

import keras.backend as K
import keras.optimizers as optimizers

import tensorflow as tf

import losses
import model
import pipelines
import utils


# The shape that we expect for raw images loaded from the disk.
RAW_IMAGE_SHAPE = (400, 400, 3)
# The shape of the images that are inputs to the networks.
INPUT_SHAPE = (44, 44, 1)
# Schedule to use when training the model. Each tuple contains a learning rate
# and the number of iterations to train for at that learning rate.
LR_SCHEDULE = [(0.00001, 10000)]

# Names to use for saving the initial versions of the models.
REF_INIT_NAME = "refiner_init.hd5"
DESC_INIT_NAME = "desc_init.hd5"


# Configure GPU VRAM usage.
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
g_session = tf.Session(config=tf_config)
K.tensorflow_backend.set_session(g_session)


class GanTrainer(object):
  """ Responsible for training the GAN networks. """

  def __init__(self, parser):
    """
    Args:
      parser: The CLI argument parser. """
    self.__parser = parser
    self.__args = self.__parser.parse_args()

  def __build_descrim(self):
    """ Builds the descriminator network, along with the machinery that generates
    batches for it.
    Returns:
      The descriminator model, and the label tensor for the descriminator.
    """
    # Run the refiner on all unlabled data.
    refined = self.__refiner_model(self.__gazecap_data_tensors[:1])

    left_eye_labeled = self.__personal_data_tensors[0]
    left_eye_unlabeled = self.__gazecap_data_tensors[0]

    # Shuffle the labeled and unlabled data into a single mini-batch.
    combined = tf.concat([left_eye_labeled, refined], 0)
    indices = tf.range(combined.shape[0])
    shuffled_indices = tf.random_shuffle(indices)

    combined = tf.gather(combined, shuffled_indices)

    # Build the model.
    desc_inputs = [combined, None, None, None, None]
    descriminator = model.DescriminatorNetwork(INPUT_SHAPE,
                                               data_tensors=desc_inputs)
    desc_model = descriminator.build()

    # Compute size of the labels tensor.
    labels_shape = desc_model.compute_output_shape(left_eye_labeled.get_shape())
    # Create initial labels tensor.
    labels_true = utils.make_real_labels(labels_shape)
    labels_fake = utils.make_fake_labels(labels_shape)
    labels = tf.concat([labels_true, labels_fake], 0)
    # Shuffle the labels.
    labels = tf.gather(labels, shuffled_indices)

    return desc_model, labels

  def __train_section(self, config):
    """ Trains for a number of iterations at one learning rate.
    Args:
      config: A configuration dictionary. It should contain the following items:
        learning_rate: The learning rate to train at.
        momentum: The momentum to use for training.
        iters: Number of iterations to train for.
        save_file: File to save the weights to. Will have .ref or .desc appended
                  for each model.
        ref_updates: Number of updates to perform for the refiner every iteration.
        desc_updates: Number of updates to perform for the descriminator every
                      iteration.
    Returns:
      Training loss and testing accuracy for this section. """
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    iters = config["iters"]
    save_file = config["save_file"]
    ref_updates = config["ref_updates"]
    desc_updates = config["desc_updates"]

    # We only use the left eye input for now.
    refiner_inputs = self.__gazecap_data_tensors[:1]

    print "\nTraining at %f for %d iters.\n" % (learning_rate, iters)

    # Set the optimizers.
    ref_opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
    desc_opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
    # The refiner expects its inputs to be passed as the targets.
    self.__refiner_model.compile(optimizer=ref_opt, loss=self.__loss,
                                 target_tensors=refiner_inputs)
    self.__desc_model.compile(optimizer=desc_opt, loss="binary_crossentropy",
                              target_tensors=[self.__desc_labels],
                              metrics=["accuracy"])

    training_loss = []
    testing_acc = []

    for i in range(0, iters):
      # Train the refiner model.
      history = self.__refiner_model.fit(epochs=1, steps_per_epoch=ref_updates)

      training_loss.extend(history.history["loss"])
      logging.info("Training loss: %s" % (history.history["loss"]))

      # Train the descriminator model.
      self.__desc_model.fit(epochs=1, steps_per_epoch=desc_updates)

      # Save the trained model.
      if i % 100 == 0:
        self.__refiner_model.save_weights(save_file + ".ref")
        self.__desc_model.save_weights(save_file + ".desc")

    return (training_loss, testing_acc)

  def __train_initial(self):
    """ Performs initial training on the two models. """
    logging.info("Performing initial training.")

    def all_defaults():
      """ Checks whether all relevant arguments are the default values.
      Returns:
        True if everything is set to defaults, false otherwise. """
      for name in arg_names:
        if getattr(self.__args, name) != self.__parser.get_default(name):
          # Not the default value.
          return False

        return True

    arg_names = ["initial_ref_updates", "initial_ref_lr",
                 "initial_ref_momentum", "initial_desc_updates",
                 "initial_desc_lr", "initial_desc_momentum"]

    defaults = all_defaults()
    if not defaults:
      logging.info("User has changed defaults, forcing retraining.")
    elif (os.path.exists(REF_INIT_NAME) and os.path.exists(DESC_INIT_NAME)):
      logging.info("Using saved initial models.")

      self.__refiner_model.load_weights(REF_INIT_NAME)
      self.__desc_model.load_weights(DESC_INIT_NAME)

      # We can skip all the training.
      return

    ref_updates = self.__args.initial_ref_updates
    ref_lr = self.__args.initial_ref_lr
    ref_momentum = self.__args.initial_ref_momentum
    desc_updates = self.__args.initial_desc_updates
    desc_lr = self.__args.initial_desc_lr
    desc_momentum = self.__args.initial_desc_momentum

    # We only use the left eye input for now.
    refiner_inputs = self.__gazecap_data_tensors[:1]

    # Create a pure regularization loss for initial refiner training.
    ref_loss = losses.RegularizationLoss(1.0)
    # Compile the refiner.
    ref_opt = optimizers.SGD(lr=ref_lr, momentum=ref_momentum)
    self.__refiner_model.compile(optimizer=ref_opt, loss=ref_loss,
                                 target_tensors=refiner_inputs)

    # Compile the descriminator.
    desc_opt = optimizers.SGD(lr=desc_lr, momentum=desc_momentum)
    self.__desc_model.compile(optimizer=desc_opt, loss="binary_crossentropy",
                              target_tensors=[self.__desc_labels],
                              metrics=["accuracy"])

    # Perform the initial refiner updates.
    self.__refiner_model.fit(epochs=1, steps_per_epoch=ref_updates)
    # Perform the initial descriminator updates.
    self.__desc_model.fit(epochs=1, steps_per_epoch=desc_updates)

    if defaults:
      # Save the initial versions.
      logging.info("Saving initial models.")

      self.__refiner_model.save_weights(REF_INIT_NAME)
      self.__desc_model.save_weights(DESC_INIT_NAME)

  def train_gan(self):
    """ Initializes and performs the entire training procedure. """
    # Build input pipelines.
    builder = pipelines.PipelineBuilder(RAW_IMAGE_SHAPE, INPUT_SHAPE[:2],
                                        self.__args.batch_size)
    # For now, our personal dataset has head pose included, but our unlabled data
    # doesn't.
    personal_input_tensors = \
        builder.build_pipeline(self.__args.personal_train_set,
                               self.__args.personal_test_set,
                               has_pose=True)
    gazecap_input_tensors = \
        builder.build_pipeline(self.__args.gazecap_train_set,
                               self.__args.gazecap_test_set)
    # Discard regression labels, as those are irrelevant for this task.
    self.__personal_data_tensors = personal_input_tensors[:5]
    self.__gazecap_data_tensors = gazecap_input_tensors[:4]
    # For GazeCapture, we don't have a pose input, so add a placeholder.
    self.__gazecap_data_tensors += (None,)

    # Build the refiner model.
    refiner = model.RefinerNetwork(INPUT_SHAPE,
                                   data_tensors=self.__gazecap_data_tensors)
    self.__refiner_model = refiner.build()

    # Create descriminator model.
    self.__desc_model, self.__desc_labels = self.__build_descrim()

    # Create loss for refiner network.
    self.__loss = losses.CombinedLoss(self.__desc_model,
                                      self.__args.reg_scale)

    # Create a coordinator and run queues.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=g_session)

    # Create configuration dict for training.
    base_config = {"momentum": self.__args.momentum,
                   "save_file": self.__args.output,
                  "ref_updates": self.__args.ref_updates,
                  "desc_updates": self.__args.desc_updates}

    # Perform initial training.
    self.__train_initial()

    # Train the model.
    for lr, iters in LR_SCHEDULE:
      config = base_config.copy()
      config["learning_rate"] = lr
      config["iters"] = iters

      self.__train_section(config)

    coord.request_stop()
    coord.join(threads)
