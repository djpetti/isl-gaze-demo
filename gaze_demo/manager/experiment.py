import signal
import sys

import menu
import params


class Experiment(object):
  """ Facilitates user control of deep learning experiments by providing a CLI
  that can be used to evaluate the model and set hyperparameters during
  training. """

  def __init__(self, testing_interval, hyperparams=None):
    """
    Args:
      testing_interval: How many training iterations to run for every testing
                        iteration.
      hyperparams: Optional custom hyperparameters to use. """
    # Whether we want to enter the menu as soon as we can.
    self.__enter_menu = False
    self.__testing_interval = testing_interval

    # Create hyperparameters.
    self.__params = hyperparams
    if self.__params is None:
      self.__params = params.HyperParams()

    # Register the signal handler.
    signal.signal(signal.SIGINT, self.__handle_signal)

    # Create the menu tree.
    self.__menus = menu.MenuTree()
    main_menu = menu.MainMenu(self.__params)
    adjust_menu = menu.AdjustMenu(self.__params)
    self.__menus.add_menu(main_menu)
    self.__menus.add_menu(adjust_menu)

  def __handle_signal(self, signum, frame):
    """ Handles the user hitting Ctrl+C. This is supposed to bring up the menu.
    Args:
      signum: The signal number that triggered this.
      frame: Current stack frame. """
    if self.__enter_menu:
      # Already entering the menu.
      return

    # Give some immediate feedback.
    print "Signal caught, entering menu after current iteration."

    # Indicate that we want to enter the menu on the next iteration.
    self.__enter_menu = True

  def __show_main_menu(self):
    """ Show the main menu. """
    self.__menus.show("main")

  def _run_training_iteration(self):
    """ Runs a single training iteration. This is meant to be overidden by a
    subclass. """
    raise NotImplementedError( \
        "_run_training_iteration() must by implemented by subclass.")

  def _run_testing_iteration(self):
    """ Runs a single testing iteration. This is meant to be overidden by a
    subclass. """
    raise NotImplementedError( \
        "_run_training_iteration() must by implemented by subclass.")

  def train(self):
    """ Runs the training procedure to completion. """
    while True:
      # Run training and testing iterations.
      for i in range(0, self.__testing_interval):
        if self.__enter_menu:
          # Show the menu.
          self.__show_main_menu()
          self.__enter_menu = False

        self._run_training_iteration()

      self._run_testing_iteration()

  def get_params(self):
    """
    Returns:
      The hyperparameters being used for this experiment. """
    return self.__params
