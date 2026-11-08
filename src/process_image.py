from enum import StrEnum
from pathlib import Path

from ultralytics import *

class ImageProcessor:
    def __init__(self) -> None:
        self._SRC: Path = Path(__file__).parent.parent
        self._DATA:Path = self._SRC / "data"

        model_path: Path = self._DATA / "yolo26n.pt"

        self._model = YOLO(model_path)

    def process_image(self, location: str) -> dict:
        results = self._model("room.png", classes = [0], conf = 0.25)
        count = len(results[0].boxes)

        data: dict = {
            EnumType.LOCATION: location,
            EnumType.COUNT: count
        }

        return data

class EnumType(StrEnum):
    LOCATION = "location"
    COUNT = "count"
