"""Select devices and capture concurrent audio and evenly spaced webcam frames."""

from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import soundfile as sf


def _sounddevice():
    # PortAudio initializes hardware on import; help and offline analysis need none.
    import sounddevice
    return sounddevice


def list_cameras() -> list[dict]:
    """Linux/macOS camera names without opening or recording from devices."""
    if sys.platform == "darwin":
        try:
            import AVFoundation
        except ImportError:
            raise RuntimeError(
                "macOS camera discovery needs PyObjC. Reinstall this project with "
                "`.venv/bin/python -m pip install -e .`."
            ) from None
        # Match OpenCV's AVFoundation backend: video + muxed devices, sorted
        # by unique ID. system_profiler/ffmpeg listing order is not sufficient.
        devices = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
        devices += list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeMuxed))
        return [{"device": index, "name": str(device.localizedName())}
                for index, device in enumerate(sorted(devices, key=lambda d: str(d.uniqueID())))]
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
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        if isinstance(camera, str):
            match = re.fullmatch(r"/dev/video(\d+)", str(Path(camera).resolve()))
            if match:
                # OpenCV 5's V4L2 backend accepts device indices, not paths.
                source, backend = int(match[1]), cv2.CAP_V4L2
        self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened():
            self.close()
            hint = " Allow Camera access for your terminal or IDE in System Settings → Privacy & Security, then retry." if sys.platform == "darwin" else ""
            raise RuntimeError(f"Cannot open webcam {camera!r}.{hint}")
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
        self._samples[self._position:self._position + count] = samples[:count]
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
    temp_dir: Path | str | None = None,
) -> tuple[Path, list[Path]]:
    """Capture one window and return (audio_path, image_paths).

    Creates a unique capture-* directory inside the project's data/temp (or
    temp_dir), containing audio.wav and image_001.png, etc. WAV embeds the
    sample rate. Files remain available until the caller deletes the directory.
    This function performs no scheduling, analysis, or uploading.
    """
    listener = Listener(resolve_microphone(camera, audio_device), sample_rate)
    with Photographer(camera) as photographer:
        photographer.take_picture()  # Discard startup frame before recording.
        audio, images = capture_sample(photographer, listener, duration, image_count)

    root = Path(temp_dir) if temp_dir is not None else Path(__file__).resolve().parents[1] / "data/temp"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="capture-", dir=root))
    try:
        audio_path = directory / "audio.wav"
        sf.write(audio_path, audio, listener.sample_rate, subtype="FLOAT")
        image_paths = []
        for index, image in enumerate(images, start=1):
            path = directory / f"image_{index:03d}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not save image: {path}")
            image_paths.append(path)
        return audio_path, image_paths
    except BaseException:
        # Remove only the new, incomplete capture directory created by this call.
        shutil.rmtree(directory)
        raise
