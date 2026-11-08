# Capture audio and images

Use Python 3.11+ and run from the repository root. Install the capture dependencies in your
Python environment:

```bash
python -m pip install numpy opencv-python sounddevice soundfile
```

Linux also needs PortAudio (`portaudio` on Arch, `libportaudio2` on Ubuntu/Debian).
Close any other app using the webcam.

## 1. Capture once

```python
from src.get_data import get_data

audio_path, image_paths = get_data(camera=4)
```

Replace `4` with your webcam index. This call blocks for about **10 seconds**,
records audio and images together, then releases the camera and microphone.
It returns one `pathlib.Path` for the WAV and an ordered list of **20** image
paths. It does not run YOLO, calculate sound levels, schedule itself, or send data.

## 2. Use the files

Each call creates its own directory under the repository's `data/temp`:

```text
data/temp/capture-<unique-id>/
    audio.wav
    image_001.png
    image_002.png
    ...
    image_020.png
```

Returned paths are absolute. Images are numbered in capture order and spread
across the 10-second window (about two per second).

```python
import cv2
import soundfile as sf

audio, sample_rate = sf.read(audio_path, dtype="float32")
# audio: mono floating-point samples; sample_rate: Hz stored in the WAV.
# Duration in seconds: len(audio) / sample_rate.

for image_path in image_paths:
    image = cv2.imread(str(image_path))  # BGR channel order; requests 640 x 480.
    # Run your model here. YOLO also accepts str(image_path) directly.
```

The WAV stores floating-point samples without integer quantization; PNGs are
lossless. The microphone's gain affects sample levels. Audio is uncalibrated.

## 3. Delete after processing

```python
import shutil

# Only after all consumers have finished using this capture:
shutil.rmtree(audio_path.parent)
```

Files are ignored by Git. They do **not** expire automatically. Each call gets a
new folder so it cannot overwrite another sample still being processed. Failed
file writes remove that call's incomplete folder and raise an exception.

## Choose devices or adjust capture

```python
from src.get_data import list_cameras, list_microphones

print(list_cameras())
print(list_microphones())

audio_path, image_paths = get_data(
    camera=4,          # Also accepts /dev/video4 or a /dev/v4l/by-id/... symlink.
    audio_device=6,    # Microphone ID or name from list_microphones().
    duration=10,
    image_count=20,
    # sample_rate=48000,       # Default: microphone's native sample rate.
    # temp_dir="/another/path",  # Default: this repository's data/temp.
)
```

On Linux, omitting `audio_device` matches the selected USB webcam's microphone.
If it cannot find one unique match, pass an input ID/name explicitly. Use
`audio_device="default"` to deliberately use the system default microphone.
On other platforms, choose an explicit microphone and camera index; camera name
listing and automatic USB matching are Linux-specific. IDs may change when
devices are reconnected. Device lists may include camera metadata nodes that
cannot capture pictures.

Call `get_data()` once per minute from your own scheduler. The 10 seconds spent
capturing count toward that minute: sleeping 60 seconds after each call would
give roughly 70 seconds between starts. Capture failures raise exceptions;
handle those in your main loop if you want retries.

## Test without hardware

```bash
python -m unittest discover -s testing -p 'test_get_data.py' -v
```
