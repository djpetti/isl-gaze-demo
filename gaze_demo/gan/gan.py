import model
import pipelines


# The shape that we expect for raw images loaded from the disk.
RAW_IMAGE_SHAPE = (400, 400, 3)
# The shape of the images that are inputs to the networks.
INPUT_SHAPE = (40, 40, 1)
# Schedule to use when training the model. Each tuple contains a learning rate
# and the number of iterations to train for at that learning rate.
LR_SCHEDULE = [(1000, 0.001)]


def build_descrim_batches(self, labeled_inputs, unlabeled_inputs, refiner):
  """ Builds the part of the graph that generates mini-batches for the
  descriminator network.
  Args:
    labeled_inputs: The output tensors from the pipeline for labeled data.
    unlabeled_inputs: The output tensors from the pipeline for unlabeled data.
    refiner: The refiner model. We need to run this to generate batch images.
  Returns:
    The batch tensor, and the label tensor for the descriminator.
  """
  # Run the refiner on all unlabled data.
  refined = refiner(unlabeled_inputs)

  # Create initial labels tensor.
  labels_true = tf.ones(labeled_inputs.shape[0])
  labels_fake = tf.zeros(unlabeled_inputs.shape[0])
  labels = tf.concat([labels_true, labels_fake], 0)

  # Shuffle the labeled and unlabled data into a single mini-batch.
  combined = tf.concat([labeled_inputs, refined], 0)
  indices = tf.range(combined.shape[0])
  shuffled_indices = tf.random_shuffle(indices)

  combined = tf.gather(combined, shuffled_indices)
  labels = tf.gather(combined, shuffled_indices)

  return combined, labels

def train_section(refine_model, desc_model, labels, loss, config):
  """ Trains for a number of iterations at one learning rate.
  Args:
    refine_model: The refiner model to train.
    desc_model: The descriminator model to train.
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

  print "\nTraining at %f for %d iters.\n" % (learning_rate, iters)

  # Set the optimizers.
  ref_opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
  desc_opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
  refine_model.compile(optimizer=ref_opt, loss=loss,
                       metrics=[model.passthrough_loss])
  desc_model.compile(optimizer=desc_opt, loss="binary_crossentropy",
                     target_tensors=[labels])

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
    ref_model.save_weights(save_file + ".ref")
    desc_model.save_weights(save_file + ".desc")

  return (training_loss, testing_acc)

def train_gan(args):
  """ Initializes and performs the entire training procedure.
  Args:
    args: The command-line arguments passed in by the user. """
  # Build input pipelines.
  builder = pipelines.PipelineBuilder(RAW_IMAGE_SHAPE, INPUT_SHAPE[:2],
                                      args.batch_size)
  personal_input_tensors = builder.build_pipeline(args.personal_train_set,
                                                  args.personal_test_set)
  gazecap_input_tensors = builder.build_pipeline(args.gazecap_train_set,
                                                 args.gazecap_test_set)
  personal_data_tensors = personal_input_tensors[:5]
  gazecap_data_tensors = gazecap_input_tensors[:5]

  # Build the refiner model.
  refiner = model.RefinerNetwork(input_shape,
                                 data_tensors=gazecap_data_tensors)
  refiner_model = refiner.build()

  # Create batch mixing subgraph for the descriminator inputs.
  desc_inputs, desc_labels = build_descrim_batches(personal_data_tensors,
                                                   gazecap_data_tensors,
                                                   refiner_model)

  descriminator = model.DescriminatorNetwork(input_shape,
                                             data_tensors=desc_inputs)
  desc_model = descriminator.build()

  # Create loss for refiner network.
  refiner_loss = model.AdversarialLoss(desc_model, args.reg_scale)

  # Create configuration dict for training.
  base_config = {"momentum": args.momentum, "save_file": args.output,
                 "ref_updates": args.ref_updates,
                 "desc_updates": args.desc_updates}

  # Train the model.
  for lr, iters in LR_SCHEDULE:
    config = base_config[:]
    config["learning_rate"] = lr
    config["iters"] = iters

    train_section(refiner_model, desc_model, desc_labels, refiner_loss, config)
