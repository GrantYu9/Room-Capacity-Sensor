"""Select devices and capture concurrent audio and evenly spaced webcam frames."""

import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def _sounddevice():
    # PortAudio initializes hardware on import; help and offline analysis need none.
    import sounddevice
    return sounddevice


def list_cameras() -> list[dict]:
    """Linux camera names without opening/recording from the devices."""
    return [
        {"device": f"/dev/{path.name}", "name": (path / "name").read_text().strip()}
        for path in sorted(Path("/sys/class/video4linux").glob("video*"))
    ]


def list_microphones() -> list[dict]:
    return [
        {"device": index, "name": info["name"], "sample_rate": info["default_samplerate"]}
        for index, info in enumerate(_sounddevice().query_devices()) if info["max_input_channels"] > 0
    ]


def _usb_parent(path: Path) -> Path | None:
    for parent in path.resolve().parents:
        if (parent / "idVendor").exists() and (parent / "idProduct").exists():
            return parent
    return None


def resolve_microphone(camera: int | str, device: int | str | None) -> int | str | None:
    """Match a Linux USB webcam to its ALSA input; never guess another mic."""
    if device == "default":
        return None
    if device is not None:
        return device
    camera_name = f"video{camera}" if isinstance(camera, int) else Path(camera).resolve().name
    usb = _usb_parent(Path("/sys/class/video4linux") / camera_name)
    candidates = []
    if usb is not None:
        cards = {
            int(card.name[4:]) for card in Path("/sys/class/sound").glob("card[0-9]*")
            if _usb_parent(card) == usb
        }
        for mic in list_microphones():
            match = re.search(r"\(hw:(\d+),\d+\)", mic["name"])
            if match and int(match[1]) in cards:
                candidates.append(mic["device"])
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Cannot uniquely match this webcam's microphone. Use --list-devices and "
        "choose --audio-device ID (audio_device=ID in Python), or explicitly "
        "use --audio-device default."
    )


class Photographer:
    def __init__(self, camera: int | str = 0):
        source = camera
        backend = cv2.CAP_ANY
        if isinstance(camera, str):
            match = re.fullmatch(r"/dev/video(\d+)", str(Path(camera).resolve()))
            if match:
                # OpenCV 5's V4L2 backend accepts device indices, not paths.
                source, backend = int(match[1]), cv2.CAP_V4L2
        self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open webcam {camera!r}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def take_picture(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam stopped delivering frames")
        return frame

    def capture(self, duration: float, image_count: int) -> list[np.ndarray]:
        # Continuously drain the camera so a later sample isn't a buffered old frame.
        start = time.monotonic()
        frames = []
        while time.monotonic() - start < duration:
            frame = self.take_picture()
            elapsed = time.monotonic() - start
            if len(frames) < image_count and elapsed >= len(frames) * duration / image_count:
                frames.append(frame.copy())
        if len(frames) != image_count:
            raise RuntimeError(f"Captured {len(frames)}/{image_count} images; lower --images")
        return frames

    def close(self):
        self._capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Listener:
    def __init__(self, device=None, sample_rate: int | None = None):
        self._sd = _sounddevice()
        info = self._sd.query_devices(device, kind="input")
        self.sample_rate = sample_rate or int(info["default_samplerate"])
        self.device = device
        self._sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=self.sample_rate)

    def start(self, duration: float):
        sample_count = round(duration * self.sample_rate)
        if sample_count < 1:
            raise ValueError("Recording duration must include at least one audio sample")
        self._samples = np.empty((sample_count, 1), dtype=np.float32)
        self._position = 0
        self._done = threading.Event()
        self._errors = []
        self._stream = self._sd.InputStream(
            device=self.device, samplerate=self.sample_rate, channels=1,
            dtype="float32", callback=self._callback,
        )
        try:
            self._stream.start()
        except BaseException:
            self.close()
            raise

    def _callback(self, samples, frames, timing, status):
        if status:
            self._errors.append(str(status))
        count = min(frames, len(self._samples) - self._position)
        # Some host APIs deliver float samples slightly outside nominal full
        # scale. Keep captured audio in the normalized range used by analysis.
        self._samples[self._position:self._position + count] = np.clip(samples[:count], -1.0, 1.0)
        self._position += count
        if self._position == len(self._samples):
            self._done.set()
            raise self._sd.CallbackStop

    def finish(self, timeout: float = 3) -> np.ndarray:
        try:
            if not self._done.wait(timeout):
                raise RuntimeError("Microphone did not deliver the full recording")
            if self._errors:
                raise RuntimeError("Audio capture was incomplete: " + "; ".join(self._errors))
            return self._samples
        finally:
            self.close()

    def close(self):
        stream = getattr(self, "_stream", None)
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
                self._stream = None


def capture_sample(photographer: Photographer, listener: Listener, duration=10.0, image_count=20):
    if not np.isfinite(duration) or duration <= 0 or image_count < 1:
        raise ValueError("duration and image_count must be positive")
    listener.start(duration)
    try:
        images = photographer.capture(duration, image_count)
        audio = listener.finish()
        return audio, images
    finally:
        listener.close()


def get_data(
    camera: int | str = 0,
    audio_device: int | str | None = None,
    *,
    duration: float = 10.0,
    image_count: int = 20,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Capture one window and return audio samples, BGR frames, and sample rate.

    Capture data stays in memory and is returned directly to the caller.
    """
    listener = Listener(resolve_microphone(camera, audio_device), sample_rate)
    with Photographer(camera) as photographer:
        photographer.take_picture()  # Discard startup frame before recording.
        audio, images = capture_sample(photographer, listener, duration, image_count)

    return audio, images, listener.sample_rate
