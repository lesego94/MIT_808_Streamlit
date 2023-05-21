from ultralytics import YOLO


def load_model(model_path):
    model = YOLO(model_path)
    return model

# Python program to get average of a list
def Average(lst):
    result = int(sum(lst) / len(lst))
    return result