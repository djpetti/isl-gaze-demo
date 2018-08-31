#!/usr/bin/python


import argparse

from gaze_demo.gan import validate

import logging_config


def build_parser():
  """ Makes a parser for arguments passed on the command line.
  Returns:
    The parser. """
  parser = argparse.ArgumentParser(description="Validate the GAN model.")

  parser.add_argument("train_set", help="The training data TFRecords file.")
  parser.add_argument("test_set", help="The testing data TFRecords file.")
  parser.add_argument("est_model", help="The estimator model to validate with.")

  parser.add_argument("-b", "--batch_size", type=int, default=256,
                      help="The batch size to use for validation.")
  parser.add_argument("-i", "--valid_iters", type=int, default=5,
                      help="Number of iterations to validate for.")
  parser.add_argument("--iter_steps", type=int, default=50,
                      help="Number of steps in each iteration.")

  return parser

def main():
  logging_config.configure_logging()
  parser = build_parser()

  validator = validate.GanValidator(parser)
  validator.validate_baseline()


if __name__ == "__main__":
  main()
