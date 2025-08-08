# config.py - Configuration for Mutual Staring Detection System

# Video settings
VIDEO_PATH = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\VIDEOS\input_video.mp4"
OUTPUT_DIR = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\STARE_FRAMES"
LOGS_DIR = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\logs"

# Processing settings
TARGET_FPS = 2  # Process at 2 FPS for stability
STARE_DURATION_SEC = 1.0  # Minimum 1 second for staring event
CONFIDENCE_THRESHOLD = 0.5  # YOLO detection confidence

# Detection parameters
MIN_YAW_MOVEMENT = 15  # Minimum head turn in degrees
MIN_YAW_DIFFERENCE = 60  # Minimum angle difference for facing each other
MAX_YAW_DIFFERENCE = 120  # Maximum angle difference

# Feature flags
SAVE_CROPPED_FACES = True
ENABLE_LOGGING = True
SHOW_DEBUG_INFO = True

# Tracking parameters
MAX_DISAPPEARED_FRAMES = 10  # Frames before removing a track
MAX_TRACKING_DISTANCE = 100  # Max distance for track matching
