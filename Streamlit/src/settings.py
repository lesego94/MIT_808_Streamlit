from pathlib import Path
import sys
import os 


# Get the relative path of the root directory with respect to the current working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

# images
IMAGES_DIR = os.path.join(ROOT, 'lib', 'Examples', 'images')
DEFAULT_IMAGE = os.path.join(IMAGES_DIR, 'example_2.png')
DEFAULT_DETECT_IMAGE = os.path.join(IMAGES_DIR, 'croc_detected.jpeg')

# video
VIDEO_DIR = os.path.join(ROOT, 'lib', 'Examples', 'videos')
VIDEO_1_PATH = os.path.join(VIDEO_DIR, 'DJI_0411.mp4')

VIDEOS_DICT = {
    'DJI_0411': VIDEO_1_PATH
}

# model
MODEL_DIR = os.path.join(ROOT, 'lib', 'models', 'weights')
DETECTION_MODEL = os.path.join(MODEL_DIR, 'yolov8n.pt')
SEGMENTATION_MODEL = os.path.join(MODEL_DIR, 'best.pt')

# Detected/segmented image dirpath locator
DETECT_LOCATOR = 'detect'
SEGMENT_LOCATOR = 'segment'





