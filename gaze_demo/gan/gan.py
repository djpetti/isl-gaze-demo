import logging

import keras.backend as K
import keras.optimizers as optimizers

import tensorflow as tf

import losses
import model
import pipelines


# The shape that we expect for raw images loaded from the disk.
RAW_IMAGE_SHAPE = (400, 400, 3)
# The shape of the images that are inputs to the networks.
INPUT_SHAPE = (40, 40, 1)
# Schedule to use when training the model. Each tuple contains a learning rate
# and the number of iterations to train for at that learning rate.
LR_SCHEDULE = [(0.001, 1000)]


# Configure GPU VRAM usage.
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
g_session = tf.Session(config=tf_config)
K.tensorflow_backend.set_session(g_session)


def build_descrim(labeled_inputs, unlabeled_inputs, refiner):
  """ Builds the descriminator network, along with the machinery that generates
  batches for it.
  Args:
    labeled_inputs: The output tensors from the pipeline for labeled data.
    unlabeled_inputs: The output tensors from the pipeline for unlabeled data.
    refiner: The refiner model. We need to run this to generate batch images.
  Returns:
    The descriminator model, and the label tensor for the descriminator.
  """
  # Run the refiner on all unlabled data.
  refined = refiner(unlabeled_inputs[:1])

  left_eye_labeled = labeled_inputs[0]
  left_eye_unlabeled = unlabeled_inputs[0]

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
  labels_true = tf.ones(labels_shape)
  labels_fake = tf.zeros(labels_shape)
  labels = tf.concat([labels_true, labels_fake], 0)
  # Shuffle the labels.
  labels = tf.gather(labels, shuffled_indices)

  return desc_model, labels

def train_section(refine_model, desc_model, gazecap_data, labels, loss,
                  config):
  """ Trains for a number of iterations at one learning rate.
  Args:
    refine_model: The refiner model to train.
    desc_model: The descriminator model to train.
    gazecap_data: Input tensors for the refiner network.
    labels: Tensor of the labels for the descriminator model.
    loss: AdversarialLoss object to use when building optimizers.
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
  refiner_inputs = gazecap_data[:1]

  print "\nTraining at %f for %d iters.\n" % (learning_rate, iters)

  # Set the optimizers.
  ref_opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
  desc_opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
  # The refiner expects its inputs to be passed as the targets.
  refine_model.compile(optimizer=ref_opt, loss=loss,
                       metrics=[loss], target_tensors=refiner_inputs)
  desc_model.compile(optimizer=desc_opt, loss="binary_crossentropy",
                     target_tensors=[labels], metrics=["accuracy"])

  training_loss = []
  testing_acc = []

  for i in range(0, iters):
    # Train the refiner model.
    history = refine_model.fit(epochs=1, steps_per_epoch=ref_updates)

    training_loss.extend(history.history["loss"])
    logging.info("Training loss: %s" % (history.history["loss"]))

    # Train the descriminator model.
    desc_model.fit(epochs=1, steps_per_epoch=desc_updates)

    # Save the trained model.
    refine_model.save_weights(save_file + ".ref")
    desc_model.save_weights(save_file + ".desc")

  return (training_loss, testing_acc)

def train_initial(refine_model, desc_model, gazecap_data, labels, args):
  """ Performs initial training on the two models.
  Args:
    refine_model: The refiner model to train.
    desc_model: The descriminator model to train.
    gazecap_data: Input tensors for the refiner network.
    labels: Tensor of the labels for the descriminator model.
    args: Parsed CLI arguments. """
  logging.info("Performing initial training.")

  ref_updates = args.initial_ref_updates
  ref_lr = args.initial_ref_lr
  ref_momentum = args.initial_ref_momentum
  desc_updates = args.initial_desc_updates
  desc_lr = args.initial_desc_lr
  desc_momentum = args.initial_desc_momentum

  # We only use the left eye input for now.
  refiner_inputs = gazecap_data[:1]

  # Create a pure regularization loss for initial refiner training.
  ref_loss = losses.RegularizationLoss(1.0)
  # Compile the refiner.
  ref_opt = optimizers.SGD(lr=ref_lr, momentum=ref_momentum)
  refine_model.compile(optimizer=ref_opt, loss=ref_loss,
                       target_tensors=refiner_inputs)

  # Compile the descriminator.
  desc_opt = optimizers.SGD(lr=desc_lr, momentum=desc_momentum)
  desc_model.compile(optimizer=desc_opt, loss="binary_crossentropy",
                     target_tensors=[labels])

  # Perform the initial refiner updates.
  refine_model.fit(epochs=1, steps_per_epoch=ref_updates)
  # Perform the initial descriminator updates.
  desc_model.fit(epochs=1, steps_per_epoch=desc_updates, metrics=["accuracy"])

def train_gan(args):
  """ Initializes and performs the entire training procedure.
  Args:
    args: The command-line arguments passed in by the user. """
  # Build input pipelines.
  builder = pipelines.PipelineBuilder(RAW_IMAGE_SHAPE, INPUT_SHAPE[:2],
                                      args.batch_size)
  # For now, our personal dataset has head pose included, but our unlabled data
  # doesn't.
  personal_input_tensors = builder.build_pipeline(args.personal_train_set,
                                                  args.personal_test_set,
                                                  has_pose=True)
  gazecap_input_tensors = builder.build_pipeline(args.gazecap_train_set,
                                                 args.gazecap_test_set)
  # Discard regression labels, as those are irrelevant for this task.
  personal_data_tensors = personal_input_tensors[:5]
  gazecap_data_tensors = gazecap_input_tensors[:4]
  # For GazeCapture, we don't have a pose input, so add a placeholder.
  gazecap_data_tensors += (None,)

  # Build the refiner model.
  refiner = model.RefinerNetwork(INPUT_SHAPE,
                                 data_tensors=gazecap_data_tensors)
  refiner_model = refiner.build()

  # Create descriminator model.
  desc_model, desc_labels = build_descrim(personal_data_tensors,
                                          gazecap_data_tensors,
                                          refiner_model)

  # Create loss for refiner network.
  refiner_loss = losses.CombinedLoss(desc_model, args.reg_scale)

  # Create a coordinator and run queues.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=g_session)

  # Create configuration dict for training.
  base_config = {"momentum": args.momentum, "save_file": args.output,
                 "ref_updates": args.ref_updates,
                 "desc_updates": args.desc_updates}

  # Perform initial training.
  train_initial(refiner_model, desc_model, gazecap_data_tensors, desc_labels,
                args)

  # Train the model.
  for lr, iters in LR_SCHEDULE:
    config = base_config.copy()
    config["learning_rate"] = lr
    config["iters"] = iters

    train_section(refiner_model, desc_model, gazecap_data_tensors, desc_labels,
                  refiner_loss, config)

  coord.request_stop()
  coord.join(threads)
