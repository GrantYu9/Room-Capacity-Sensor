from enum import StrEnum
import librosa
from pathlib import Path
import numpy as np

from ultralytics import YOLO

class ImageProcessor:
    def __init__(self) -> None:
        self._ROOT: Path = Path(__file__).parent.parent
        self._DATA:Path = self._ROOT / "data"

        model_path: Path = self._DATA / "external" / "yolo26n.pt"

        self._model = YOLO(model_path)

    def process_data(self, audio: Path, image: Path, location: str) -> dict:
        """Processes data.

        Args:
            audio: Path of the audio file to be processed.
            image: Path of the image to be processed.
            location: Location of the Pi.
        
        Returns:
            Dictionary holding data.
        """

        data: dict = {
            EnumType.LOCATION: location
        }

        data += self._room_level_db(audio)
        data += self._process_image(image)

        return data


    def _process_image(self, image: Path) -> dict:
        results = self._model(image, classes = [0], confidence = 0.25)
        count = len(results[0].boxes)

        data: dict = {
            EnumType.COUNT: count
        }

        return data

    def _room_level_db(path: Path) -> int:
        data, _ = librosa.load(path, sr=None, mono=True)
        rms = np.sqrt(np.mean(data ** 2))

        decibels:int = int(20 * np.log10(rms) if rms > 0 else float("-inf"))


        data: dict = {
            EnumType.DECIBELS: decibels
        }

        return data

class EnumType(StrEnum):
    COUNT = "count"
    DECIBELS = "decibels"
    LOCATION = "location"