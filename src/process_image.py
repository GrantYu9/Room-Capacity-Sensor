from ultralytics import *
from enum import Enum

model = YOLO('yolo26n.pt')

results = model("room.png", classes = [0], conf = 0.25)

count = len(results[0].boxes)

class Capacity(Enum):
    LOCATION = "Study Room 1"
    COUNT = count

