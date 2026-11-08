"""Local detection previews preserve raw frames and aggregate-only uploads."""

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from ultralytics.engine.results import Results

from src.process_data import ImageProcessor
from src.upload import reading_payload


def detector(frame, counts):
    processor = ImageProcessor.__new__(ImageProcessor)
    processor._confidence = 0.15
    processor._model = Mock()
    processor._model.predict.side_effect = [
        [Results(orig_img=frame.copy(), path="frame.png", names={0: "person"},
                 boxes=np.array([[30, 35, 100, 90, 0.8, 0]] * count, dtype=np.float32).reshape(-1, 6))]
        for count in counts
    ]
    return processor


def test_person_box_and_label_saved_separately_without_modifying_original(tmp_path):
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    original = tmp_path / "image_001.png"
    assert cv2.imwrite(str(original), frame)
    original_bytes = original.read_bytes()
    processor = detector(frame, [1])
    output = tmp_path / "annotated/image_001.jpg"
    result = processor._process_image(original, annotation_path=output)
    assert result["count"] == 1
    assert result["average_confidence"] == pytest.approx(0.8)
    assert result["annotated_image_file"] == str(output.resolve())
    saved = cv2.imread(str(output))
    assert saved.shape == frame.shape
    assert np.any(saved != frame)
    assert original.read_bytes() == original_bytes


def test_every_frame_gets_an_ordered_preview_including_empty_detections(tmp_path):
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    processor = detector(frame, [1, 0, 2])
    result = processor.process_data(np.zeros(16000, dtype=np.float32), [frame] * 3,
                                    sample_rate=16000, annotation_dir=tmp_path / "annotated")
    expected = [str((tmp_path / f"annotated/image_{i:03d}.jpg").resolve()) for i in range(1, 4)]
    assert result["annotated_image_files"] == expected
    assert all(Path(path).is_file() for path in expected)
    assert result["per_image_counts"] == [1, 0, 2]
    assert result["count"] == 1
    assert result["average_confidence"] == pytest.approx(0.8)
    payload = reading_payload(result, "room-1", "2026-11-08T12:00:00Z")
    assert set(payload) == {"room_id", "people_count", "average_confidence", "noise_level", "noise_unit", "measured_at"}
    assert "annotated" not in str(payload)


def test_offline_analysis_does_not_save_without_an_output_directory():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    processor = detector(frame, [1])
    with patch("src.process_data.cv2.imwrite") as save:
        result = processor.process_data(np.zeros(16000, dtype=np.float32), [frame], sample_rate=16000)
    save.assert_not_called()
    assert result["annotated_image_files"] == []


def test_failed_annotation_write_is_reported(tmp_path):
    processor = detector(np.zeros((120, 160, 3), dtype=np.uint8), [1])
    with patch("src.process_data.cv2.imwrite", return_value=False), pytest.raises(OSError, match="annotated image"):
        processor._process_image("frame.png", annotation_path=tmp_path / "annotated/image_001.jpg")
