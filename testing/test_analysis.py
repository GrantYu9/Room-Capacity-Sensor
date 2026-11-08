"""Hardware-free regression tests for analysis, sampling, and scheduling."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from interface import main as app
from src import get_data
from src.process_data import ImageProcessor


class DetectionBoxes(list):
    def __init__(self, count):
        super().__init__([None] * count)
        self.conf = np.full(count, 0.8)


class AnalysisTests(unittest.TestCase):
    def make_processor(self, counts):
        # Avoid loading weights; tests below exercise the real analysis methods.
        processor = ImageProcessor.__new__(ImageProcessor)
        processor._confidence = 0.25
        processor._model = Mock()
        processor._model.predict.side_effect = [
            [SimpleNamespace(boxes=DetectionBoxes(count))] for count in counts
        ]
        return processor

    def test_every_image_contributes_including_zero_and_fractional_mean(self):
        processor = self.make_processor([0, 1, 3])
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        result = processor.process_data(np.full(100, 0.5), images, "test", sample_rate=10)
        self.assertEqual(result["per_image_counts"], [0, 1, 3])
        self.assertAlmostEqual(result["average_confidence"], 0.8)
        self.assertIsNone(result["per_image_average_confidences"][0])
        np.testing.assert_allclose(result["per_image_average_confidences"][1:], [0.8, 0.8])
        self.assertEqual(result["count"], 1)
        self.assertIs(type(result["count"]), int)
        self.assertAlmostEqual(result["mean_count"], 4 / 3)
        self.assertAlmostEqual(result["decibels"], -6.020599913)
        self.assertEqual(result["audio_duration_seconds"], 10)
        self.assertEqual(processor._model.predict.call_count, 3)
        for call in processor._model.predict.call_args_list:
            self.assertEqual(call.kwargs["classes"], [0])
            self.assertEqual(call.kwargs["conf"], 0.25)

    def test_sound_level_averages_energy_instead_of_decibel_values(self):
        processor = self.make_processor([0])
        result = processor.process_data(np.array([0.0, 1.0]), "test.jpg", sample_rate=2)
        self.assertAlmostEqual(result["decibels"], -3.010299957)

    def test_window_confidence_is_weighted_by_detection_count(self):
        processor = self.make_processor([])
        processor._process_image = Mock(side_effect=[
            {"count": 1, "average_confidence": 0.2},
            {"count": 0, "average_confidence": None},
            {"count": 3, "average_confidence": 0.9},
        ])
        result = processor.process_data(np.zeros(10), ["x.jpg"] * 3, sample_rate=10)
        self.assertAlmostEqual(result["average_confidence"], 0.725)

    def test_counts_round_half_up_and_keep_raw_mean(self):
        for counts, expected in [([0, 1], 1), ([1, 2], 2), ([0, 0, 1], 0), ([2, 2, 3], 2)]:
            result = self.make_processor(counts).process_data(np.zeros(10), ["x.jpg"] * len(counts), sample_rate=10)
            self.assertEqual(result["count"], expected)
            self.assertIs(type(result["count"]), int)
            self.assertAlmostEqual(result["mean_count"], np.mean(counts))

    def test_silence_is_finite_json(self):
        result = self.make_processor([0]).process_data(np.zeros(20), "test.jpg", sample_rate=2)
        self.assertEqual(result["decibels"], -120)
        self.assertIsNone(result["average_confidence"])
        json.dumps(result, allow_nan=False)

    def test_missing_or_invalid_input_fails_instead_of_reporting_zero(self):
        for audio, images in [(np.array([]), ["x.jpg"]), (np.array([np.nan]), ["x.jpg"]),
                              (np.ones(10), []), (np.ones(10, dtype=int), ["x.jpg"])]:
            with self.subTest(audio=audio, images=images), self.assertRaises(ValueError):
                self.make_processor([]).process_data(audio, images, sample_rate=10)


class CaptureTests(unittest.TestCase):
    def test_linux_camera_path_is_passed_as_index(self):
        with patch.object(get_data.cv2, "VideoCapture") as capture:
            with get_data.Photographer("/dev/video4"):
                pass
        capture.assert_called_once_with(4, get_data.cv2.CAP_V4L2)
        capture.return_value.release.assert_called_once()

    def test_webcam_microphone_is_matched_by_usb_card(self):
        with patch.object(get_data, "_usb_parent", return_value=Path("/usb/webcam")), \
             patch.object(Path, "glob", return_value=[Path("/sys/class/sound/card3")]), \
             patch.object(get_data, "list_microphones", return_value=[
                 {"device": 4, "name": "Built-in (hw:1,0)"},
                 {"device": 6, "name": "Brio 101: USB Audio (hw:3,0)"},
             ]):
            self.assertEqual(get_data.resolve_microphone(4, None), 6)

    def make_listener(self):
        class CallbackStop(Exception):
            pass
        backend = Mock()
        backend.CallbackStop = CallbackStop
        backend.query_devices.return_value = {"default_samplerate": 10}
        with patch.object(get_data, "_sounddevice", return_value=backend):
            listener = get_data.Listener(6)
        return listener, backend

    def test_audio_trims_final_block_to_exact_duration(self):
        listener, backend = self.make_listener()
        listener.start(0.5)
        listener._callback(np.ones((4, 1), dtype=np.float32), 4, None, None)
        with self.assertRaises(backend.CallbackStop):
            listener._callback(np.ones((4, 1), dtype=np.float32) * 2, 4, None, None)
        np.testing.assert_array_equal(listener.finish().ravel(), [1, 1, 1, 1, 2])
        backend.InputStream.return_value.close.assert_called_once()

    def test_audio_overflow_is_an_error(self):
        listener, backend = self.make_listener()
        listener.start(0.5)
        with self.assertRaises(backend.CallbackStop):
            listener._callback(np.ones((5, 1)), 5, None, "input overflow")
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            listener.finish()
        backend.InputStream.return_value.close.assert_called_once()

    def test_missing_audio_times_out_and_releases_device(self):
        listener, backend = self.make_listener()
        listener.start(0.5)
        with self.assertRaisesRegex(RuntimeError, "full recording"):
            listener.finish(timeout=0)
        backend.InputStream.return_value.close.assert_called_once()

    def test_camera_failure_always_stops_audio(self):
        listener = Mock()
        photographer = Mock()
        photographer.capture.side_effect = RuntimeError("unplugged")
        with self.assertRaisesRegex(RuntimeError, "unplugged"):
            get_data.capture_sample(photographer, listener)
        listener.start.assert_called_once_with(10.0)
        listener.close.assert_called_once()

    def test_images_are_spread_across_capture_window(self):
        clock = [0.0]
        def read():
            clock[0] += 0.025
            return True, np.full((1, 1, 3), clock[0])
        photographer = get_data.Photographer.__new__(get_data.Photographer)
        photographer._capture = Mock()
        photographer._capture.read.side_effect = read
        with patch.object(get_data.time, "monotonic", side_effect=lambda: clock[0]):
            frames = photographer.capture(1, 4)
        timestamps = [float(frame[0, 0, 0]) for frame in frames]
        np.testing.assert_allclose(timestamps, [0, 0.25, 0.5, 0.75], atol=0.026)
        self.assertGreaterEqual(clock[0], 1)

    def test_microphone_selection_never_falls_back_silently(self):
        self.assertEqual(get_data.resolve_microphone(4, 6), 6)
        self.assertIsNone(get_data.resolve_microphone(4, "default"))
        with patch.object(get_data, "_usb_parent", return_value=None):
            with self.assertRaisesRegex(ValueError, "--audio-device"):
                get_data.resolve_microphone(4, None)


class FileCaptureTests(unittest.TestCase):
    def test_capture_files_round_trip_and_do_not_overwrite_previous_calls(self):
        audio = np.full((1000, 1), 0.125, dtype=np.float32)
        images = [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(20)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(get_data, "resolve_microphone", return_value=6), \
             patch.object(get_data, "Listener") as listener, \
             patch.object(get_data, "Photographer"), \
             patch.object(get_data, "capture_sample", return_value=(audio, images)):
            listener.return_value.sample_rate = 100
            audio_path, image_paths = get_data.get_data(camera=4, temp_dir=directory)
            second_audio, _ = get_data.get_data(camera=4, temp_dir=directory)
            self.assertNotEqual(audio_path.parent, second_audio.parent)
            self.assertEqual(audio_path.parent.parent, Path(directory).resolve())
            saved_audio, rate = sf.read(audio_path, always_2d=True)
            self.assertEqual(rate, 100)
            self.assertEqual(len(saved_audio) / rate, 10)
            np.testing.assert_array_equal(saved_audio, audio)
            self.assertEqual(len(image_paths), 20)
            for path, expected in zip(image_paths, images):
                self.assertEqual(path.parent, audio_path.parent)
                np.testing.assert_array_equal(get_data.cv2.imread(str(path)), expected)

    def test_failed_save_removes_only_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(get_data, "resolve_microphone", return_value=6), \
             patch.object(get_data, "Listener") as listener, \
             patch.object(get_data, "Photographer"), \
             patch.object(get_data, "capture_sample", return_value=(np.zeros((10, 1)), [np.zeros((8, 8, 3))])), \
             patch.object(get_data.cv2, "imwrite", return_value=False):
            listener.return_value.sample_rate = 10
            previous = Path(directory) / "previous.txt"
            previous.write_text("keep")
            with self.assertRaisesRegex(OSError, "Could not save image"):
                get_data.get_data(camera=4, temp_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()), [previous])
            self.assertEqual(previous.read_text(), "keep")


class SchedulerTests(unittest.TestCase):
    def assert_window_starts(self, processing_seconds, expected):
        clock = [0.0]
        starts = []
        def capture(*args, **kwargs):
            starts.append(clock[0])
            if len(starts) == 3:
                raise KeyboardInterrupt
            clock[0] += 10
            return Path("audio.wav"), [Path("frame.png")]
        def analyze(*args, **kwargs):
            clock[0] += processing_seconds
            return {"count": 0}
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(app, "ImageProcessor") as processor, \
             patch.object(app, "get_data", side_effect=capture), \
             patch.object(app.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(app.time, "sleep", side_effect=sleep), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            processor.return_value.process_data.side_effect = analyze
            self.assertEqual(app.main([]), 0)
        self.assertEqual(starts, expected)

    def test_windows_start_every_minute_without_processing_drift(self):
        self.assert_window_starts(5, [0, 60, 120])

    def test_slow_inference_skips_missed_windows(self):
        self.assert_window_starts(65, [0, 120, 240])

    def test_invalid_settings_are_rejected(self):
        for args in [["--duration", "nan"], ["--images", "0"], ["--interval", "5"],
                     ["--confidence", "2"], ["--sample-rate", "0"]]:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(args)


if __name__ == "__main__":
    unittest.main()
