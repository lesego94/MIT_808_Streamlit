from pathlib import Path
import time
import streamlit as st
import torch
import cv2
import PIL
import os
import settings
import helper
import tempfile
import pandas as pd
from PIL import Image

import numpy as np
import piexif


# Streamlit title for the dashboard
st.title("Crocodile Monitoring Dashboard")

# Create a sidebar header titled 'Settings'
st.sidebar.header("Settings")

# Sidebar option for selecting the task, it can be 'Detection' or 'Individual Detection'
mlmodel_radio = st.sidebar.radio(
    "Select Task", ['Detection','Individual Detection (Pending)' ])

# Sidebar slider for adjusting the model's confidence level
conf = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Set up paths for models based on the selected task
if mlmodel_radio == 'Individual Detection (Pending)':
    dirpath_locator = settings.DETECT_LOCATOR
    model_path = Path(settings.DETECTION_MODEL)
elif mlmodel_radio == 'Detection':
    dirpath_locator = settings.SEGMENT_LOCATOR
    model_path = Path(settings.SEGMENTATION_MODEL)
# Load the selected model, catch and print exceptions if model fails to load
try:
    model = helper.load_model(model_path)
except Exception as ex:
    print(ex)
    st.write(f"Unable to load model. Check the specified path: {model_path}")

source_img = None
# Sidebar header for image/video configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)


# If image is selected
if source_radio == settings.IMAGE:
    # Upload an image of specified formats
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2) 

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption='Default Image',
                     use_column_width=True)
        else:
            image = PIL.Image.open(source_img)
            # Create a temporary file with .jpg extension
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                temp_path = f.name
            # Save the uploaded image to this temporary file
            image.save(temp_path)                
            
            st.image(source_img, caption='Uploaded Image',
                     use_column_width=True)
            
    

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            with torch.no_grad():
                res = helper.inference(temp_path,conf,source_img)
                os.unlink(temp_path)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
        
                No_crocs = res[0].boxes.shape[0]
                st.write("Number of Crocodiles",No_crocs)               
                

                
elif source_radio == settings.VIDEO:
    

    source_vid = st.sidebar.file_uploader("Upload a Video", type = ("mp4"),accept_multiple_files=False)
    
    if source_vid is None:
        source_vid = open(settings.VIDEO_1_PATH, 'rb')
        st.video(source_vid)
        filename = settings.VIDEO_1_PATH

        
    else:

        st.video(source_vid)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_vid.read())
        filename = tfile.name
        
    No_crocs = []
        
    if st.sidebar.button('Detect Video Objects'):
        vid_cap = cv2.VideoCapture(filename)
        stframe = st.empty()
        while (vid_cap.isOpened()):
            latest_iteration = st.empty()
            
            success, image = vid_cap.read()
            if success:
                
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=conf)
                res_plotted = res[0].plot()
                stframe.image(res_plotted,
                              caption='Detected Video',
                              channels="BGR",
                              use_column_width=True)
            
                No_crocs.append(res[0].boxes.shape[0])
                
             
                Ave_Croc = helper.Average(No_crocs)
                Message = "Number of crocodiles: " + str(Ave_Croc)
                latest_iteration.write(Message)
                time.sleep(1)
                latest_iteration.empty()


    
 # Initialize the dataframe
df = pd.DataFrame(columns=['ID','Date','Image', 'Time', 'Latitude', 'Longitude'])
   
if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = df
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0


 



    
# Check if an image has been uploaded
if source_img is not None:
    # If an image has been uploaded, create the 'Add row' button
    if st.button('Add row'):
        Lat, Long = helper.get_coordinates(image)        

        # Specify the values you want to append
        new_row = {
            'ID': st.session_state['counter'],
            'Date':helper.get_Date(),
            'Image': source_img.name,
            'Description': ' ',
            'Time': helper.get_SA_Time(),
            'Latitude': Lat ,
            'Longitude': Long,
            'No of Crocodiles':3
        }    


        # Append the new row to the DataFrame    
        st.session_state['dataframe'] = pd.concat([st.session_state['dataframe'], pd.DataFrame([new_row])], ignore_index=True)
    
    # Increment the counter
    st.session_state['counter'] += 1

    edited_df = st.experimental_data_editor(st.session_state['dataframe'], use_container_width=True,num_rows= "dynamic")

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(edited_df)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Record.csv',
            mime='text/csv',
        )
        
    with col2:
        if st.button('Clear'):
            st.session_state.dataframe = pd.DataFrame(columns=['ID', 'Image','Description', 'Time', 'Latitude', 'Longitude','No of Crocodiles'])
            st.session_state.counter = 0





