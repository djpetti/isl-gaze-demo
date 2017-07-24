import random

import numpy as np

import config

def _unit_vector(vector):
  """ Converts a vector to a unit vector.
  Args:
    vector: The vector to get the unit vector of.
  Returns:
    The corresponding unit vector. """
  return vector / np.linalg.norm(vector)

def _angle_between(v1, v2):
  """ Computes the angle between two vectors, in radians.
  Args:
    v1: The first vector.
    v2: The second vector.
  Returns:
    The angle between the two vectors. """
  v1_u = _unit_vector(v1)
  v2_u = _unit_vector(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def _compute_face_size():
  """ Computes the size of a random simulated user's face.
  Returns:
    The width and height of the face, in cm. """
  # The base face size.
  mean_width = config.Synthetic.FACE_WIDTH
  mean_height = config.Synthetic.FACE_HEIGHT

  # Add noise.
  width = np.random.normal(mean_width, config.Synthetic.FACE_STDDEV)
  height = np.random.normal(mean_height, config.Synthetic.FACE_STDDEV)

  return (width, height)

def _compute_view_bounds(fov, screen_size, z_dist, cam_offset):
  """ Computes how far the camera can see in one direction.
  Args:
    fov: The FOV of the camera in that direction.
    screen_size: The size of the screen in that direction.
    z_dist: How far away we are from the camera.
    cam_offset: How far the camera is from the center of the screen in this
                dimension.
  Returns:
    The mininum and maximum centimeter coordinates that the camera can see to.
  """
  cam_view = z_dist * np.tan(fov / 2)
  max_cam = (screen_size / 2) + cam_view
  min_cam = (screen_size / 2) - cam_view

  # Account for offset.
  min_cam += cam_offset
  max_cam += cam_offset

  return (min_cam, max_cam)

def _compute_head_pos_bounds(gaze_vec, z_dist, min_distance, max_gaze,
                             face_size, dimension):
  """ Computes the maximum and minimum head positions in one dimension.
  Args:
    gaze_vec: The gaze vector.
    z_dist: An initial distance for the head to be from the camera.
    min_distance: The minimum distance the head can be from the camera.
    max_gaze: The maximum number of cm in this dimension that the gaze can go
              while remaining on-screen.
    face_size: The size of the face, along this dimension.
    dimension: The dimension to compute for. This is the index of that dimension
               in the gaze vector.
  Returns:
    The minimum, maximum and maximum values, and the new z_dist we have to use
    to make this work. It will return (None, None, None) if it can't find a
    solution. """
  fov = config.Camera.CAMERA_FOV[dimension]

  # Project gaze vector onto the xz or yz plane.
  gaze_planar = np.array([gaze_vec[dimension], gaze_vec[2]])
  # Angle between the gaze vector and the screen plane.
  gaze_screen_angle = _angle_between(gaze_planar, np.array([1, 0]))

  while True:
    # Figure out where our head can be to keep the gaze in-frame.
    max_head_gaze = max_gaze - z_dist / np.tan(gaze_screen_angle)
    min_head_gaze = -(max_gaze - max_head_gaze)

    # Figure out the edges of what the camera can see at this distance.
    camera_pos = config.Camera.CAMERA_POS[dimension]
    min_cam, max_cam = _compute_view_bounds(fov, max_gaze, z_dist, camera_pos)

    # Figure out where our head must be to be completely visible.
    max_head_cam = max_cam - (face_size / 2)
    min_head_cam = min_cam + (face_size / 2)

    # Use this data to find the global range of the head position.
    max_head = min(max_head_gaze, max_head_cam)
    min_head = max(min_head_gaze, min_head_cam)

    if max_head < min_head:
      # We're in an unsolvable situation. Try moving the head inward and see if
      # we can fix it.
      z_dist /= 2.0
      if z_dist < min_distance:
        # There's probably now way to make this work. Give up.
        return (None, None, None)

    else:
      # It's good.
      return (min_head, max_head, z_dist)

def _generate_face_mask(head_pos, head_size, head_pose, fov):
  """ Generates a simulated mask image.
  Args:
    head_pos: The position of the head in 3D space. ([x, y, z])
    head_size: The width and height of the head, in cm.
    head_pose: The attitude of the head. ([pitch, yaw, roll])
    fov: The horizontal and vertical FOVs of the camera, in radians.
  Returns:
    Two points, in the form of ((x1, y1), (x2, y2)), which define the face mask.
  """
  face_width, face_height = head_size
  x_pos, y_pos, z_pos = head_pos
  h_fov, v_fov = fov
  pitch, yaw, roll = head_pose
  print "Head pos: %f, %f, %f" % (x_pos, y_pos, z_pos)
  print "Head pose: %f, %f, %f" % (pitch, yaw, roll)
  print "Face size: %f, %f" % (face_width, face_height)

  # Compute the size of the observed camera frame.
  cam_pos_x, cam_pos_y = config.Camera.CAMERA_POS
  view_min_x, view_max_x = _compute_view_bounds(h_fov, config.SCREEN_WIDTH_CM,
                                                z_pos, cam_pos_x)
  view_min_y, view_max_y = _compute_view_bounds(v_fov, config.SCREEN_HEIGHT_CM,
                                                z_pos, cam_pos_y)
  print "View dims: (%f, %f), (%f, %f)" % (view_min_x, view_min_y, view_max_x,
                                           view_max_y)
  view_width = view_max_x - view_min_x
  view_height = view_max_y - view_min_y

  # Compute the size of the face in the camera frame.
  image_face_width = face_width * config.Camera.FOCAL_LENGTH / z_pos
  image_face_height = face_height * config.Camera.FOCAL_LENGTH / z_pos

  # Compute the position of the face in the frame.
  face_x_pos = (x_pos - view_min_x) / view_width
  face_y_pos = (y_pos - view_min_y) / view_height

  # Compute size change due to head rotation.
  image_face_width = image_face_width * np.sin(np.pi / 2.0 - yaw)
  image_face_height = image_face_height * np.sin(np.pi / 2.0 - pitch)
  # We're going to assume roll doesn't affect things.

  # Put it all together.
  x_min = face_x_pos - image_face_width / 2
  x_max = face_x_pos + image_face_width / 2
  y_min = face_y_pos - image_face_height / 2
  y_max = face_y_pos + image_face_height / 2

  return ((x_min, y_min), (x_max, y_max))


def _compute_random_head_pos(gaze_vec, head_pose, face_size):
  """ Computes a random head position that is still visible to the (simulated)
  camera, and still within screen bounds.
  Args:
    gaze_vec: The gaze vector, in the form [x, y, z].
    head_pose: The head pose, in the form [pitch, yaw, roll].
    face_size: The size of the face, in cm, in the form (width, height).
  Returns:
    A tuple containing the location of the head in 3D space in the form
    [x, y, z]. All three elements will be None if this condition is unsolvable. """
  face_width, face_height = face_size

  # Bound the distance from the camera by when we fill up the frame.
  fov_w, fov_h = config.Camera.CAMERA_FOV
  min_distance_h = (face_height / 2) / np.tan(fov_h / 2)
  min_distance_w = (face_width / 2) / np.tan(fov_w / 2)
  min_distance = max(min_distance_w, min_distance_h)

  # Start by choosing a random distance from the camera.
  max_distance = config.Camera.MAX_CAMERA_DIST
  z_dist = random.uniform(min_distance, max_distance)

  # Compute head pos boundaries for x and y dimensions.
  min_x, max_x, z_dist = _compute_head_pos_bounds(gaze_vec, z_dist,
                                                  min_distance,
                                                  config.SCREEN_WIDTH_CM,
                                                  face_width, 0)
  if z_dist == None:
    # Unsolvable.
    return (None, None, None)
  min_y, max_y, z_dist = _compute_head_pos_bounds(gaze_vec, z_dist,
                                                  min_distance,
                                                  config.SCREEN_HEIGHT_CM,
                                                  face_height, 1)
  if z_dist == None:
    return (None, None, None)

  # Pick x and y positions for the head.
  x_dist = random.uniform(min_x, max_x)
  y_dist = random.uniform(min_y, max_y)

  return (x_dist, y_dist, z_dist)

def compute_gaze_and_head_box(gaze_vec, head_pose):
  """ Given a synthetic image, chooses a random simulated position for the head
  in space, and computes the resulting gaze point.
  Returns:
    The gaze point on the screen, as well as the head bounding box, both in
    screen units, or (None, None) if unsolvable. """
  # Determine randomized face size.
  face_size = _compute_face_size()

  # Determine a position for the head.
  head_x, head_y, head_z = _compute_random_head_pos(gaze_vec, head_pose,
                                                    face_size)
  if not head_x:
    # Unsolvable.
    return (None, None)

  # Compute the actual gaze point based on the head position.
  gaze_vec_xz = np.array([gaze_vec[0], gaze_vec[2]])
  gaze_vec_yz = np.array([gaze_vec[1], gaze_vec[2]])
  # Angle between the gaze vector and the screen plane.
  gaze_angle_x = _angle_between(gaze_vec_xz, np.array([1, 0]))
  gaze_angle_y = _angle_between(gaze_vec_yz, np.array([1, 0]))

  # These are the offsets due to head pose.
  gaze_offset_x = head_z / np.tan(gaze_angle_x)
  gaze_offset_y = head_z / np.tan(gaze_angle_y)

  gaze_x = head_x + gaze_offset_x
  gaze_y = head_y + gaze_offset_y

  # Convert to screen units.
  gaze_x /= config.SCREEN_WIDTH_CM
  gaze_y /= config.SCREEN_HEIGHT_CM

  # Compute what the face would actually look like in the frame.
  fov_w, fov_h = config.Camera.CAMERA_FOV
  mask_points = _generate_face_mask((head_x, head_y, head_z),
                                    face_size,
                                    head_pose, (fov_w, fov_h))

  return ((gaze_x, gaze_y), mask_points)
