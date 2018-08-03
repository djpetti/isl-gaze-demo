import cv2


def reshape_image(image, shape, offset=(0, 0)):
  """ Reshapes a stored image so that it is a consistent shape and size.
  Args:
    image: The image to reshape.
    shape: The shape we want the image to be.
    offset: An optional offset. This can be used to direct it not to crop to the
            center of the image. In the tuple, the horizontal offset comes
            before the vertical one.
  Returns:
    The reshaped image.
  """
  # Crop the image to just the center square.
  if len(image.shape) == 3:
    # It may have multiple color channels.
    height, width, _ = image.shape
  else:
    height, width = image.shape

  target_width, target_height = shape

  # Find the largest we can make the initial crop.
  multiplier = 1
  if width > target_width:
    multiplier = width / target_width
  elif height > target_height:
    multiplier = height / target_height
  target_width *= multiplier
  target_height *= multiplier

  crop_width = target_width
  crop_height = target_height
  # Our goal here is to keep the same aspect ratio as the original.
  if width <= target_width:
    # We need to reduce the width for our initial cropping.
    crop_width = width
    crop_height = target_height * (float(crop_width) / target_width)
  if height <= target_height:
    # We need to reduce the height for our initial cropping.
    crop_height = height
    crop_width = target_width * (float(crop_height) / target_height)

  crop_width = int(crop_width)
  crop_height = int(crop_height)

  # Crop the image.
  crop_left = (width - crop_width) / 2
  crop_top = (height - crop_height) / 2

  # Account for the crop offset.
  offset_left, offset_top = offset
  crop_left += offset_left
  crop_top += offset_top
  # Bound it in the image.
  crop_left = max(0, crop_left)
  crop_left = min(width - 1, crop_left)
  crop_top = max(0, crop_top)
  crop_top = min(height - 1, crop_top)

  image = image[crop_top:(crop_height + crop_top),
                crop_left:(crop_width + crop_left)]

  # Set a proper size, which should just be directly scaling up or down.
  image = cv2.resize(image, shape)

  return image
