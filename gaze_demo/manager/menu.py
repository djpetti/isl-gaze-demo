import sys


class Menu(object):
  """ Represents a menu. """

  # Inidcates that we want to leave the menu system.
  CONTINUE = "ContinueTraining"

  def __init__(self, name, params):
    """
    Args:
      name: The name of the menu.
      params: The hyperparameters that we can update with the menu. """
    self.__name = name
    self._params = params

    # No options currently in the menu.
    self.__options = {}

  def __print_menu(self):
    """ Prints the menu for the user to see. """
    # Create header line.
    header = "%s Menu:" % (self.__name)
    header = header.title()
    print header

    # Display the options alphabetically.
    option_names = self.__options.keys()
    option_names.sort()
    for option in option_names:
      desc, command = self.__options[option]
      print "\t%s: %s" % (option, desc)

  def add_option(self, option, desc, command):
    """ Adds a new option to the menu.
    Args:
      option: The option that the user can type.
      desc: The description of what the option does.
      command: The function to call if the option is chosen. This function
               should takes the option name as an argument, and should return
               the name of the menu to go to after the option is selected.
               To stay on the same menu, the name of this menu should be
               returned. If the special value Menu.CONTINUE is returned
               instead, it will leave the menu system and continue training. """
    self.__options[option] = (desc, command)

  def remove_option(self, option):
    """ Removes an option from the menu.
    Args:
      option: The option to remove. """
    self.__options.pop(option)

  def update_description(self, option, desc):
    """ Updates an option description.
    Args:
      option: The option to update.
      desc: The new description. """
    _, command = self.__options[option]
    self.__options[option] = (desc, command)

  def show(self):
    """ Shows the menu, and waits for the user to select something.
    Returns:
      The return value of the command callback. """
    # Display the menu.
    self.__print_menu()

    # Wait for input.
    selection = None
    while selection not in self.__options:
      selection = raw_input("(Choose an option): ")

    # Perform the command.
    _, command = self.__options[selection]
    return command(selection)

  def get_name(self):
    """
    Returns:
      The name of the menu. """
    return self.__name

class MenuTree(object):
  """ Encapsulates a tree of menus that are traversable by the user. """

  def __init__(self):
    # The internal menus, organized by name.
    self.__menus = {}
    # Stacks the menus that were shown previously.
    self.__previous_menus = []

  def __previous_menu(self, *args):
    """ Goes to the previous menu in the tree. """
    assert len(self.__previous_menus) >= 2

    # First, we're going to have to pop off the current menu.
    self.__previous_menus.pop()
    # Now, get the previous menu.
    menu_name = self.__previous_menus.pop()

    # Indicate that we should go to it.
    return menu_name

  def add_menu(self, menu):
    """ Adds a new menu to the tree.
    Args:
      menu: The menu to add. """
    name = menu.get_name()
    self.__menus[name] = menu

  def show(self, name):
    """ Shows a particular menu and waits for user input. """
    if name not in self.__menus:
      raise NameError("No such menu '%s'." % (name))
    menu = self.__menus[name]

    added_back_option = False
    if len(self.__previous_menus) != 0:
      # Add an option for going back to the previous menu.
      menu.add_option("b", "Back", self.__previous_menu)
      added_back_option = True

    # Push this menu onto the stack.
    self.__previous_menus.append(menu.get_name())

    # Display the menu and wait for the user.
    next_menu_name = menu.show()
    if next_menu_name is None:
      # Invalid name.
      raise ValueError("Menu option callback must return a menu name.")

    # Remove the back option.
    if added_back_option:
      menu.remove_option("b")

    if next_menu_name == Menu.CONTINUE:
      # Special value indicating that we want to continue training.
      self.__previous_menus.pop()
      return
    if next_menu_name == menu.get_name():
      # If we stay on the same menu, don't stack it multiple times.
      self.__previous_menus.pop()
    # Continue to the next menu.
    self.show(next_menu_name)


class MainMenu(Menu):
  """ Represents the main menu. """

  def __init__(self, params):
    super(MainMenu, self).__init__("main", params)

    # Adjusts a hyperparameter.
    self.add_option("a", "Adjust hyper-parameters", self.__adjust)
    # Continues training.
    self.add_option("c", "Continue training", self.__continue)
    # Exits the training program.
    self.add_option("q", "Exit", self.__exit)

  def __exit(self, *args):
    """ Halts training and exits the program. """
    sys.exit(0)

  def __continue(self, *args):
    """ Exits the menu and continues training. """
    return Menu.CONTINUE

  def __adjust(self, *args):
    """ Goes to the hyperparameter adjustment menu. """
    return "adjust"

class AdjustMenu(Menu):
  """ Represents the hyperparameter adjustment menu. """

  def __init__(self, params):
    super(AdjustMenu, self).__init__("adjust", params)

    self.__used_names = set()
    # Maps option names to hyperparameter names.
    self.__option_params = {}

    # Add options to adjust all hyperparameters.
    param_names = self._params.get_all()
    for name in param_names:
      option = self.__unique_shorthand(name)
      # Create a description that includes the current value.
      desc = self.__make_description(name)

      self.add_option(option, desc, self.__adjust_param)
      self.__option_params[option] = name

  def __make_description(self, param_name):
    """ Creates a description for a parameter based on its name and value.
    Args:
      name: The name of the parameter.
    Returns:
      The description. """
    value = self._params.get_value(param_name)
    return "%s (Currently %s)" % (param_name, str(value))

  def __unique_shorthand(self, name):
    """ Creates a unique shorthand for an option.
    Args:
      name: The name to create a shorthand for.
    Returns:
      The shorthand it created. """
    shorthand = name[0]

    for i in range(1, len(name)):
      if shorthand not in self.__used_names:
        # This is unique.
        self.__used_names.add(shorthand)
        return shorthand

      # Add another letter.
      shorthand += name[i]

    raise ValueError("Duplicate param '%s'?" % (name))

  def __adjust_param(self, option):
    """ Adjusts a hyperparameter value. """
    # Get the name of the parameter.
    name = self.__option_params[option]

    # Ask the user for a new value.
    value = input("Enter value for %s: " % (name))
    self._params.update(name, value)

    # Update the description with the new value.
    desc = self.__make_description(name)
    self.update_description(option, desc)

    # Stay on the same menu.
    return self.get_name()
