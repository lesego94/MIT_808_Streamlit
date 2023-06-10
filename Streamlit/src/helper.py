import datetime
import cv2
import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ExifTags
import settings
import time
import random


def load_model(model_path):
    model = YOLO(model_path)
    return model

# Python program to get average of a list
def Average(lst):
    result = int(sum(lst) / len(lst))
    return result


def get_SA_Time(): 
    # Get the current time
    current_time = datetime.datetime.now()

    # Define the GMT+2 offset
    gmt_offset = datetime.timedelta(hours=2)

    # Calculate the GMT+2 time
    gmt_plus_2_time = current_time + gmt_offset

    # Format and display the GMT+2 time
    formatted_time = gmt_plus_2_time.strftime("%H:%M:%S")
    
    return str(formatted_time)

def get_Date(): 
    # Get the current time
    current_time = datetime.datetime.now()

    # Define the GMT+2 offset
    gmt_offset = datetime.timedelta(hours=2)

    # Calculate the GMT+2 time
    gmt_plus_2_time = current_time + gmt_offset

    # Format and display the GMT+2 time
    formatted_time = gmt_plus_2_time.strftime("%Y-%m-%d")
    
    return str(formatted_time)


def get_coordinates(img): 
    try:    
        exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
        
        if 'GPSInfo' not in exif:
            return "None", "None"
        
        if 1 not in exif['GPSInfo'] or 2 not in exif['GPSInfo'] or 3 not in exif['GPSInfo'] or 4 not in exif['GPSInfo']:
            return "Incomplete GPSInfo in EXIF data", "Incomplete GPSInfo in EXIF data"
        
        
        Lat = f"{exif['GPSInfo'][1]}: {', '.join(map(str, exif['GPSInfo'][2]))}"
        Long= f"{exif['GPSInfo'][3]}: {', '.join(map(str, exif['GPSInfo'][4]))}"

        return Lat, Long
    except AttributeError:
        return "No EXIF data", "No EXIF data"
    
    


def create_video_from_image(image_path, video_name):
    # load the image
    frame = cv2.imread(image_path)
    Video_path = settings.VIDEO_P2_PATH
    # get the dimensions of the image
    height, width, layers = frame.shape

    # initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # be sure to use lowercase

    video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

    # write the image to the video the specified number of times
    for i in range(5):
        video.write(frame)

    # close the video file
    video.release()


def inference(image,conf):    
    model = load_model(settings.SEGMENTATION_MODEL)
    res = model.predict(source = image,conf=conf, project = settings.IMAGE_SEGMENTS,save_crop =True)
    return res

