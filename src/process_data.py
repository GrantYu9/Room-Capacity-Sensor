from enum import StrEnum
from pathlib import Path
import subprocess

import librosa
import numpy as np
from ultralytics import YOLO

class Processor:
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

        temp: dict = self._room_level_db(audio)
        data[EnumType.DECIBELS] = temp[EnumType.DECIBELS]

        temp: dict = self._process_image(image)
        data[EnumType.COUNT] = temp[EnumType.COUNT]

        return data


    def _process_image(self, image: Path) -> dict:
        results = self._model(image, classes = [0], conf = 0.25)
        count = len(results[0].boxes)

        data: dict = {
            EnumType.COUNT: count
        }

        return data

    def _room_level_db(self, path: Path) -> int:
        subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", path.with_suffix(".wav")], check=True)

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
