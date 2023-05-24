import datetime

from ultralytics import YOLO


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