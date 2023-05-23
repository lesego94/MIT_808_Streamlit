from pathlib import Path
import time
import streamlit as st
import torch


import os
import settings
import helper
import tempfile
import pandas as pd

from PIL import Image
from moviepy.editor import VideoFileClip
import numpy as np



# Sidebar
st.title("Crocodile Monitoring Dashboard")

st.sidebar.header("Settings")

mlmodel_radio = st.sidebar.radio(
    "Select Task", ['Detection','Individual Detection (Pending)' ])
conf = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
if mlmodel_radio == 'Individual Detection (Pending)':
    dirpath_locator = settings.DETECT_LOCATOR
    model_path = Path(settings.DETECTION_MODEL)
elif mlmodel_radio == 'Detection':
    dirpath_locator = settings.SEGMENT_LOCATOR
    model_path = Path(settings.SEGMENTATION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    print(ex)
    st.write(f"Unable to load model. Check the specified path: {model_path}")

source_img = None
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)


# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    save_radio = st.sidebar.radio("Save image to download", ["Yes", "No"])
    save = True if save_radio == 'Yes' else False
    col1, col2 = st.columns(2)

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption='Default Image',
                     use_column_width=True)
        else:
            image = PIL.Image.open(source_img)
            st.image(source_img, caption='Uploaded Image',
                     use_column_width=True)
            
    

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                with torch.no_grad():
                    res = model.predict(
                        source = image, save=save, save_txt=save, exist_ok=True, conf=conf, project =f"runs/{dirpath_locator}/predict/image.jpeg")
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                             use_column_width=True)
        
                No_crocs = res[0].boxes.shape[0]
                st.write("Number of Crocodiles",No_crocs)               
                
                

                
# elif source_radio == settings.VIDEO:
    

#     source_vid = st.sidebar.file_uploader("Upload a Video", type = ("mp4"),accept_multiple_files=False)
    
#     if source_vid is None:
#         source_vid = open('videos/DJI_0411.mp4','rb')
#         st.video(source_vid)
#         filename = 'videos/DJI_0411.mp4'

        
#     else:

#         st.video(source_vid)
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(source_vid.read())
#         filename = tfile.name
        
#     No_crocs = []
        
#     if st.sidebar.button('Detect Video Objects'):
#         vid_cap = cv2.VideoCapture(filename)
#         stframe = st.empty()
#         while (vid_cap.isOpened()):
#             latest_iteration = st.empty()
            
#             success, image = vid_cap.read()
#             if success:
                
#                 image = cv2.resize(image, (720, int(720*(9/16))))
#                 res = model.predict(image, conf=conf)
#                 res_plotted = res[0].plot()
#                 stframe.image(res_plotted,
#                               caption='Detected Video',
#                               channels="BGR",
#                               use_column_width=True)
            
#                 No_crocs.append(res[0].boxes.shape[0])
                
             
#                 Ave_Croc = helper.Average(No_crocs)
#                 Message = "Number of crocodiles: " + str(Ave_Croc)
#                 latest_iteration.write(Message)
#                 time.sleep(1)
#                 latest_iteration.empty()


elif source_radio == settings.VIDEO:
    source_vid = st.sidebar.file_uploader("Upload a Video", type=("mp4"), accept_multiple_files=False)
    
    if source_vid is None:
        source_vid = open('videos/DJI_0411.mp4','rb')
        st.video(source_vid)
        filename = 'videos/DJI_0411.mp4'
    else:
        st.video(source_vid)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_vid.read())
        filename = tfile.name

    No_crocs = []
        
    if st.sidebar.button('Detect Video Objects'):
        vid_clip = VideoFileClip(filename)
        stframe = st.empty()
        
        for i, frame in enumerate(vid_clip.iter_frames()):  
            latest_iteration = st.empty()
            
            pil_image = Image.fromarray(frame)  # Convert the image to a PIL Image object.
            pil_image = pil_image.resize((720, int(720*(9/16))), Image.ANTIALIAS)  # Resize the image.
            image = np.array(pil_image)  # Convert the image back to a numpy array.

            # MoviePy's iter_frames returns RGB frames, so no need to convert RGB to BGR.

            res = model.predict(image, conf=conf)
            res_plotted = res[0].plot()
            stframe.image(res_plotted,
                          caption='Detected Video',
                          channels="RGB",
                          use_column_width=True)
        
            No_crocs.append(res[0].boxes.shape[0])
            
            Ave_Croc = helper.Average(No_crocs)
            Message = "Number of crocodiles: " + str(Ave_Croc)
            latest_iteration.write(Message)
            time.sleep(1)
            latest_iteration.empty()

# ...






    
 # Initialize the dataframe
df = pd.DataFrame(columns=['ID','Date','Image', 'Time', 'Latitude', 'Longitude'])
   
if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = df
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0

if st.button('Add row'):
    # Specify the values you want to append
    new_row = {'ID': st.session_state['counter'],'Date':'12-04-2023','Image': source_img.name,
               'Description': 'Kruger park may 4th' ,'Time': '2023-05-21 14:20:00', 'Latitude': 12.9715987, 'Longitude': 77.5945627, 'No of Crocodiles':3}

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




    #         image = cv2.resize(image, (720, int(720*(9/16))))
    #         res = model.predict(image, conf=conf)
    #         res_plotted = res[0].plot()
    #         stframe.image(res_plotted,
    #                         caption='Detected Video',
    #                         channels="BGR",
    #                         use_column_width=True)

    #         No_crocs.append(res[0].boxes.shape[0])

    #         Ave_Croc = helper.Average(No_crocs)
    #         Message = "Number of crocodiles: " + str(Ave_Croc)
    #         latest_iteration.write(Message)
    #         time.sleep(0.05)  # Adjust sleep time to manage frame rate
    #         latest_iteration.empty()

    


# elif source_radio == settings.RTSP:
#     source_rtsp = st.sidebar.text_input("rtsp stream url")
#     if st.sidebar.button('Detect Objects'):
#         vid_cap = cv2.VideoCapture(source_rtsp)
#         stframe = st.empty()
#         while (vid_cap.isOpened()):
#             success, image = vid_cap.read()
#             if success:
#                 image = cv2.resize(image, (720, int(720*(9/16))))
#                 res = model.predict(image, conf=conf)
#                 res_plotted = res[0].plot()
#                 stframe.image(res_plotted,
#                               caption='Detected Video',
#                               channels="BGR",
#                               use_column_width=True
#                               )

# elif source_radio == settings.YOUTUBE:
#     source_youtube = st.sidebar.text_input("YouTube Video url")
#     if st.sidebar.button('Detect Objects'):
#         video = pafy.new(source_youtube)
#         best = video.getbest(preftype="mp4")
#         cap = cv2.VideoCapture(best.url)
#         stframe = st.empty()
#         while (cap.isOpened()):
#             success, image = cap.read()
#             if success:
#                 image = cv2.resize(image, (720, int(720*(9/16))))
#                 res = model.predict(image, conf=conf)
#                 res_plotted = res[0].plot()
#                 stframe.image(res_plotted,
#                               caption='Detected Video',
#                               channels="BGR",
#                               use_column_width=True
#                               )



