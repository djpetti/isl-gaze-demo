class Params(object):
  """ Superclass for collections of numeric parameters. """

  def __init__(self):
    # Actual dict containing parameters and their values.
    self.__parameters = {}
    # Keeps track of parameters that were changed for get_changed.
    self.__changed = set()

  def add(self, name, value):
    """ Adds a new hyperparameter.
    Args:
      name: The name of the parameter.
      value: The initial value of the parameter. """
    if name in self.__parameters:
      raise NameError("Parameter '%s' already exists." % (name))

    self.__parameters[name] = value
    # New parameters automatically count as "changed".
    self.__changed.add(name)

  def add_if_not_set(self, name, value):
    """ Adds a new parameter, but only if it doesn't already exist. Otherwise,
    it does nothing.
    Args:
      name: The name of the parameter.
      value: The value to set for the parameter. """
    if name in self.__parameters:
      # Already set.
      return

    self.add(name, value)

  def update(self, name, value):
    """ Updates an existing parameter.
    Args:
      name: The name of the parameter.
      value: The initial value of the parameter. """
    if name not in self.__parameters:
      raise NameError("Parameter '%s' does not exist." % (name))

    if self.__parameters[name] == value:
      # Already set. Don't update.
      return

    self.__parameters[name] = value
    # Mark the parameter as changed.
    self.__changed.add(name)

  def get_value(self, name):
    """ Gets the value of a hyperparameter.
    Args:
      name: The name of the hyperparameter.
    Returns:
      The value of said hyperparameter. """
    if name not in self.__parameters:
      raise NameError("Parameter '%s' does not exist." % (name))

    return self.__parameters[name]

  def get_all(self):
    """
    Returns:
      List of the names of all parameters. """
    return self.__parameters.keys()

  def get_changed(self):
    """
    Returns:
      List of the names of the parameters that have been updated since this
      was last called. """
    changed = list(self.__changed)
    # Clear the set for next time.
    self.__changed.clear()

    return changed


class HyperParams(Params):
  """ Represents hyperparameters that can be used to configure an experiment. """

  pass

class Status(Params):
  """ Status indicators from the experiment. """

  pass
