"""Integer person counts and qualitative microphone level analysis."""

from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from ultralytics import YOLO
from .noise import NoiseCalibration, audio_level


class EnumType(StrEnum):
    AVERAGE_CONFIDENCE = "average_confidence"
    COUNT = "count"
    DECIBELS = "decibels"
    LOCATION = "location"


class ImageProcessor:
    def __init__(self, model_path: Path | None = None, confidence: float = 0.15, noise_calibration=None):
        model_path = model_path or Path(__file__).resolve().parents[1] / "data/external/yolo26x.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be between 0 (exclusive) and 1")
        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._noise_calibration = noise_calibration or NoiseCalibration.load()

    def process_data(self, audio, image, location: str = "", *, sample_rate=None, annotation_dir=None) -> dict:
        """Analyze audio samples/a sound file and every supplied BGR image/path.

        Count rounds the all-frame arithmetic mean half up; mean_count retains it.
        Decibels is 20 log10(RMS) over all normalized audio samples (dBFS),
        with a -120 dBFS floor for silence. If annotation_dir is supplied, save
        a separate JPEG per frame with person boxes, labels and confidence.
        Original frames are unchanged. No media is uploaded.
        """
        level = audio_level(audio, sample_rate)
        calibration = getattr(self, "_noise_calibration", None) or NoiseCalibration.load()

        images = [image] if isinstance(image, (str, Path, np.ndarray)) else list(image)
        if not images:
            raise ValueError("At least one image is required")
        image_results = [
            self._process_image(frame, annotation_path=Path(annotation_dir) / f"image_{index:03d}.jpg")
            if annotation_dir is not None else self._process_image(frame)
            for index, frame in enumerate(images, start=1)
        ]
        counts = [result[EnumType.COUNT] for result in image_results]
        detections = sum(counts)
        average_confidence = (sum(result[EnumType.AVERAGE_CONFIDENCE] * result[EnumType.COUNT]
                                  for result in image_results if result[EnumType.COUNT]) / detections
                              if detections else None)
        return {
            EnumType.LOCATION: location,
            EnumType.COUNT: (2 * sum(counts) + len(counts)) // (2 * len(counts)),
            "mean_count": float(np.mean(counts)),
            EnumType.AVERAGE_CONFIDENCE: average_confidence,
            "per_image_average_confidences": [result[EnumType.AVERAGE_CONFIDENCE] for result in image_results],
            **level,
            **calibration.classify(level["decibels"]),
            "per_image_counts": counts,
            "image_count": len(counts),
            "annotated_image_files": [result["annotated_image_file"] for result in image_results
                                      if "annotated_image_file" in result],
        }

    def _process_image(self, image, *, annotation_path=None) -> dict:
        # COCO class 0 is person. Ultralytics uses `conf`, not `confidence`.
        results = self._model.predict(
            source=str(image) if isinstance(image, Path) else image,
            classes=[0], conf=self._confidence, verbose=False, save=False,
        )
        if len(results) != 1 or results[0].boxes is None:
            raise RuntimeError("YOLO did not return a detection result for the image")
        boxes = results[0].boxes
        count = len(boxes)
        result = {EnumType.COUNT: count,
                  EnumType.AVERAGE_CONFIDENCE: float(boxes.conf.mean()) if count else None}
        if annotation_path is not None:
            path = Path(annotation_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            annotated = results[0].plot(boxes=True, labels=True, conf=True)
            if not cv2.imwrite(str(path), annotated):
                raise OSError(f"Could not save annotated image: {path}")
            result["annotated_image_file"] = str(path)
        return result


class Processor(ImageProcessor):
    """Upstream schema, mono downmix, and integer dBFS compatibility."""

    def process_data(self, audio, image, location: str = "", *, sample_rate=None) -> dict:
        if isinstance(audio, (str, Path)):
            audio, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
            # The upstream librosa.load(..., mono=True) path downmixed channels
            # before RMS. Keep this legacy behavior separate from the local
            # ImageProcessor API, which measures the supplied full recording.
            audio = audio.mean(axis=1)
        result = super().process_data(audio, image, location, sample_rate=sample_rate)
        return {EnumType.LOCATION: result[EnumType.LOCATION],
                EnumType.COUNT: result[EnumType.COUNT],
                EnumType.AVERAGE_CONFIDENCE: result[EnumType.AVERAGE_CONFIDENCE],
                EnumType.DECIBELS: int(result[EnumType.DECIBELS])}
