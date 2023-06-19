import datetime
import cv2
import os
import torch
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ExifTags
import settings
import time
import random
import shutil
import sleap
import io
import matplotlib.pyplot as plt
import math
import tempfile
import io

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
    
    



def inference(image,conf,uploaded_file):   
    save_crop = Check_newfile(uploaded_file) 
    model = load_model(settings.SEGMENTATION_MODEL)
    res = model.predict(source = image,conf=conf, project = settings.IMAGE_SEGMENTS,save_crop =save_crop)
    return res

def Check_newfile(uploaded_file):  
    # Check if this is a new file
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file:
        st.session_state['last_file'] = uploaded_file
        return True
    else:
        return False
        

def sleap_predictor(image_path):
    predictor = sleap.load_model([settings.DETECTION_MODEL])
    image = sleap.load_video(image_path)
    labels = predictor.predict(image)
    # Plot the first labeled frame
    
    fig = plt.figure()
    labels[0].plot(scale=1)

    with io.BytesIO() as buf:
        plt.savefig(buf, format='png')
        buf.seek(0)
        pil_img = Image.open(buf)
        st.image(pil_img, caption='Labeled Frame')
        
    return labels

def key_table(labels):
    instance = labels[0][0]
    pts = instance.numpy()
    Pose_table = pd.DataFrame({'Key-Points': ['Snout', 'UB', 'MB', 'LB', 'UBL', 'UBR', 'LBL', 'LBR'],
                        'x-cord': [pts[0][0],pts[1][0],pts[2][0],pts[3][0],pts[4][0],pts[5][0],pts[6][0],pts[7][0]],
                        'y-cord': [pts[0][1],pts[1][1],pts[2][1],pts[3][1],pts[4][1],pts[5][1],pts[6][1],pts[7][1]]}).round(1)
    return Pose_table
        
        

def split_tif_image(source_img, desired_size=640):
    
    """Split .tif image into smaller tiles."""
    
    img = Image.open(source_img)
    width, height = img.size

    # Determine the number of chunks to split the image into
    x_chunks = math.ceil(width / desired_size)
    y_chunks = math.ceil(height / desired_size)

    # List to hold all the image chunk temporary files
    img_chunk_files = []
    #list of image coordinates
    coords = []
    # Split the image
    for i in range(y_chunks):
        row_chunk_files = []  # New list for each row of chunks
        for j in range(x_chunks):
            left = j * desired_size
            upper = i * desired_size
            right = min((j+1) * desired_size, width)
            lower = min((i+1) * desired_size, height)
            
            img_chunk = img.crop((left, upper, right, lower))
            
            # Convert the image chunk to RGB mode
            img_chunk_rgb = img_chunk.convert("RGB")
            
            # Save the image chunk to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img_chunk_rgb.save(f, "JPEG")
                row_chunk_files.append(f.name)
                coords.append((left, upper, right, lower))
        
        img_chunk_files.append(row_chunk_files)

    return img_chunk_files, width, height

def display_grid(img_chunk_files, x, y, GRID_SIZE=5):
    
    # Display the images for the current grid
    for i in range(y, min(y + GRID_SIZE, len(img_chunk_files))):
        row_chunk_files = img_chunk_files[i]
        cols = st.columns(GRID_SIZE)
        for j in range(x, min(x + GRID_SIZE, len(row_chunk_files))):
            with cols[j - x]:
                # Open the image (which is now a JPEG, not a TIFF)
                img = Image.open(row_chunk_files[j])
                
                # Convert the image to PNG format
                with io.BytesIO() as output:
                    img.save(output, format="PNG")
                    png_image = output.getvalue()
                
                # Display the PNG image in Streamlit
                st.image(png_image, use_column_width=True)


def navigate(img_chunk_files):
    GRID_SIZE = 5

    # Initialize the session state
    if 'x' not in st.session_state:
        st.session_state.x = 0
    if 'y' not in st.session_state:
        st.session_state.y = 0

    col1, col2 = st.columns(2)
    with col1: 
        # Display navigation buttons
        if st.button('Up'):
            st.session_state.y = max(0, st.session_state.y - GRID_SIZE)

        if st.button('Down'):
            st.session_state.y = min(len(img_chunk_files) - GRID_SIZE, st.session_state.y + GRID_SIZE)
            
    with col2:
        if st.button('Left'):
            st.session_state.x = max(0, st.session_state.x - GRID_SIZE)

        if st.button('Right'):
            st.session_state.x = min(len(img_chunk_files[0]) - GRID_SIZE, st.session_state.x + GRID_SIZE)
            
    display_grid(img_chunk_files, st.session_state.x, st.session_state.y)


def display_inference_grid(img_chunk_files, conf,uploaded_file):
    GRID_SIZE = 5
    x = st.session_state.x
    y = st.session_state.y

    # Display the images for the current grid after inference
    for i in range(y, min(y + GRID_SIZE, len(img_chunk_files))):
        row_chunk_files = img_chunk_files[i]
        cols = st.columns(GRID_SIZE)
        for j in range(x, min(x + GRID_SIZE, len(row_chunk_files))):
            with cols[j - x]:
                temp_path = row_chunk_files[j]
                # Run inference
                with torch.no_grad():
                    res = inference(temp_path, conf,uploaded_file)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, use_column_width=True)

                # No_crocs = res[0].boxes.shape[0]
                # st.write("Number of Crocodiles", No_crocs)



def delete_temp_files(img_chunk_files):
    for chunk_list in img_chunk_files:
        for img_file in chunk_list:
            if os.path.isfile(img_file):
                os.unlink(img_file)


#modified inference function
def inference_2(img_path, conf, uploaded_file):
    save_crop = Check_newfile(uploaded_file) 
    model = load_model(settings.SEGMENTATION_MODEL)
    
    # image = Image.open(img_path)
    image = img_path
    # Run inference on the image using your YOLO model
    res = model.predict(source = image,conf=conf, project = settings.IMAGE_SEGMENTS,save_crop =save_crop)
    boxes = res[0].boxes
    
    # Plot the result
    res_plotted = res[0].plot()[:, :, ::-1]
    
    # Save the plotted result to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        plt.imsave(f.name, res_plotted)
    
    # Return the path to the temp file
    return f.name

def process_chunks_through_yolo(img_chunk_files, conf):
    processed_chunk_files = []

    for i, row_chunk_files in enumerate(img_chunk_files):
        processed_row_chunk_files = []
        for j, chunk_file in enumerate(row_chunk_files):
            with torch.no_grad():
                res = inference2(chunk_file, conf)
            res_plotted = res[0].plot()[:, :, ::-1]

            # Save the processed image chunk to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                plt.imsave(f, res_plotted)
                processed_row_chunk_files.append(f.name)

        processed_chunk_files.append(processed_row_chunk_files)

    return processed_chunk_files


def inference2(image,conf):   

    model = load_model(settings.SEGMENTATION_MODEL)
    res = model.predict(source = image,conf=conf, project = settings.IMAGE_SEGMENTS,save_crop =True)
    return res


def stitch_processed_chunks_together(processed_chunk_files, width, height):
    stitched_image = Image.new('RGB', (width, height))

    for i, row_chunk_files in enumerate(processed_chunk_files):
        for j, chunk_file in enumerate(row_chunk_files):
            chunk_image = Image.open(chunk_file)
            stitched_image.paste(chunk_image, (j * chunk_image.width, i * chunk_image.height))

    return stitched_image
