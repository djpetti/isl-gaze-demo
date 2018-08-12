#!/usr/bin/python

import argparse
import logging

from gaze_demo.gan import gan


def configure_logging():
  """ Configure logging handlers. """
  # Cofigure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " + \
      "[%(levelname)s] %(message)s")

  stream_handler.setFormatter(formatter)

  root.addHandler(stream_handler)

def build_parser():
  """ Makes a parser for arguments passed on the command line.
  Returns:
    The parser. """
  parser = argparse.ArgumentParser(description="Train the GAN model.""")

  parser.add_argument("personal_train_set",
                      help="The path to the personal training TFRecords data.")
  parser.add_argument("personal_test_set",
                      help="The path to the personal testing TFRecords data.")
  parser.add_argument("gazecap_train_set",
      help="The path to the GazeCapture training TFRecords data.")
  parser.add_argument("gazecap_test_set",
      help="The path to the GazeCapture testing TFRecords data.")
  parser.add_argument("-o", "--output", default="gan_weights.hd5",
                      help="Output model file.")

  parser.add_argument("--learning_rate", type=float, default=0.00001,
                      help="Learning rate for training.")
  parser.add_argument("--momentum", type=float, default=0.9,
                      help="Momentum to use for training.")
  parser.add_argument("--ref_updates", type=int, default=2,
                      help="Number of times to update refiner each iteration.")
  parser.add_argument("--desc_updates", type=int, default=1,
      help="Number of times to update descriminator each iteration.")
  parser.add_argument("--reg_scale", type=float, default=1.0,
                      help="Scale factor for regularization loss.")
  parser.add_argument("--batch_size", type=int, default=256,
                      help="The batch size to use when training.")

  parser.add_argument("--initial_ref_updates", type=int, default=1000,
      help="No. of times to update the refiner initially with only reg loss.")
  parser.add_argument("--initial_desc_updates", type=int, default=200,
      help="No. of times to update the descriminator initially.")
  parser.add_argument("--initial_ref_lr", type=float, default=0.00001,
                      help="Learning rate for initial ref update.")
  parser.add_argument("--initial_desc_lr", type=float, default=0.01,
                      help="Learning rate for initial desc update.")
  parser.add_argument("--initial_ref_momentum", type=float, default=0.9,
                      help="Momentum for initial ref update.")
  parser.add_argument("--initial_desc_momentum", type=float, default=0.9,
                      help="Momentum for initial desc update.")

  return parser

def main():
  configure_logging()
  parser = build_parser()

  trainer = gan.GanTrainer(parser)
  trainer.train()


if __name__ == "__main__":
  main()
