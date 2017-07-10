# The dimensions of the user's screen, in pixels.
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1070

# Model to use for prediction.
EYE_MODEL="eye_model.hd5"

# The minimum ratio of eye height to eye width before we consider the eye
# closed.
EYE_OPEN_RATIO = 0.1
# Minimum confidence before we ignore detections.
MIN_CONFIDENCE = 0.55
