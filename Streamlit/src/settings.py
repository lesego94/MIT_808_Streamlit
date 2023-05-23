from pathlib import Path
import sys
import os 

# # Get the absolute path of the current file
# file_path = Path(__file__).resolve()

# # Get the parent directory of the current file
# root_path = file_path.parent

# # Add the root path to the sys.path list if it is not already there
# if root_path not in sys.path:
#     sys.path.append(str(root_path))

# # Get the relative path of the root directory with respect to the current working directory
# ROOT = root_path.relative_to(Path.cwd())

ROOT  = current_dir = os.getcwd() 
# ROOT = os.path.dirname(current_dir)


# Source
IMAGE = 'Image'
VIDEO = 'Video'
# RTSP = 'RTSP'
# YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO]

# images
IMAGES_DIR = f"{ROOT}/lib/Examples/images"
DEFAULT_IMAGE = f'{IMAGES_DIR}/example_2.png'
DEFAULT_DETECT_IMAGE = f'{IMAGES_DIR}/croc_detected.jpeg'

#video
VIDEO_DIR = f'{ROOT}/lib/Examples/videos'
VIDEO_1_PATH = f'{VIDEO_DIR}/DJI_0411.mp4'

VIDEOS_DICT = {
    'DJI_0411': VIDEO_1_PATH
}

# model
MODEL_DIR = f'{ROOT}/lib/models/weights'
DETECTION_MODEL = f'{MODEL_DIR}/yolov8n.pt'
SEGMENTATION_MODEL = f'{MODEL_DIR}/best.pt'


# Detected/segmented image dirpath locator
DETECT_LOCATOR = 'detect'
SEGMENT_LOCATOR = 'segment'


# Webcam
WEBCAM_PATH = 0
\


import streamlit as st
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Function to handle the GPS metadata
def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (t, val) in GPSTAGS.items():
                if t in exif[idx]:
                    geotagging[val] = exif[idx][t]

    return geotagging

# Function to handle the conversion of the coordinates
def get_decimal_from_dms(dms, ref):

    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)

# Initialize the dataframe
df = pd.DataFrame(columns=['Image', 'Timestamp', 'Latitude', 'Longitude'])

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
   
