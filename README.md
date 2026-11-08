# Room Capacity Sensor

Local webcam and microphone analysis. The first sample starts immediately; later
samples start every **60 seconds**. Each sample captures **10 seconds of audio**
and **20 images**, spread across the same window. YOLO26 runs on every image,
counts people (COCO class 0), and rounds the arithmetic mean to an integer,
including images with zero people. Half values round up; `mean_count` preserves
the fractional mean for diagnostics.

Sound level is RMS over the full recording, converted with `20 * log10(RMS)` to
**dBFS**. This is relative to the microphone's digital full scale, not calibrated
dBA or sound pressure level. A versioned noise profile maps dBFS to the website's
Quiet/Moderate/Lively bands through a relative index in [0,1]. The bundled profile
uses generated WAV fixtures and remains provisional for real microphones.
Silence uses a finite -120 dBFS floor. Microphone gain/automatic gain control
affects these values. See [noise calibration](docs/noise-calibration.md).

Without `--upload`, analysis runs locally. With `--upload`, each completed window
is queued durably and sent immediately through the website's authenticated API;
only aggregate metadata is uploaded. No PostgreSQL credentials belong here.
Each capture saves WAV/PNGs in a unique `data/temp/capture-*` directory. By
default, the latest window stays available until the next window finishes
processing, including across restarts and `--once` runs. A failed capture leaves
the previous window intact. Use `--keep-captures` to keep every window instead.

## Connect to the website

For interactive setup, run from this directory:

```sh
.venv/bin/python interface/setup.py
```

Choose a detected webcam, then fuzzy-search website rooms by ID, name, building
or address. The picker uses `fzf` when installed; `--no-fzf` selects the numbered
search fallback. Linux metadata-only video nodes are excluded without recording
video. The webcam's USB microphone is matched automatically when possible;
otherwise choose an input from the list.

Setup reuses a matching `.env.*` device profile; `--env .env.laptop` selects a
particular profile. A different room needs a newly registered device token. If
the trusted adjacent app checkout is available, setup offers to register one
there, preserving existing devices and credentials. Otherwise provide a private
profile provisioned through the backend. Database credentials stay in the app.

The resulting `run-sensor.local.sh` contains hardware choices and the profile
path, never the token. Setup asks before starting capture; `--configure-only`
only saves that launcher. To reuse it or test one window:

```sh
./run-sensor.local.sh
./run-sensor.local.sh --once
```

The launcher and private profiles are ignored by Git.

Follow [room/device mapping](docs/room-device-mapping.md) to register an installation
in the app backend and securely copy its private credentials to `.env.sensor`.
Deploy the updated API and apply `npm run db:confidence` in the app repository first.

```sh
.venv/bin/python interface/main.py --upload --env .env.sensor \
  --camera 0 --audio-device default
```

The config requires `API_URL`, `ROOM_ID`, `DEVICE_ID`, `DEVICE_TOKEN`; environment
variables take precedence. HTTPS is required outside loopback development.
The API resolves identity from the token; the sensor checks both device/room in
the acknowledgement. A label supplied via `--location` never changes the mapping.
`data/outbox/readings.sqlite3` keeps failed readings across restarts, with no token
or media. Each window attempts up to five queued readings, newest first, and
retries transient failures up to three times using the original timestamp/data.
The recurring loop logs upload failures and continues; `--once` returns failure
when the upload stays queued. Wrong credentials, rooms and conflicting duplicate
timestamps require operator attention. Keep the outbox when upgrading.

## Setup

Run from this directory with Python 3.11+:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

On Linux, install PortAudio through your system package manager if the audio
library cannot load (`portaudio` on Arch, `libportaudio2` on Debian/Ubuntu).
The selected camera and microphone must be accessible and free for capture.

The sensor defaults to **YOLO26x**, a confidence threshold of **0.15**, and a
requested camera resolution of **1920×1080** (actual output depends on the
camera). Download the official weights once from this directory:

```sh
.venv/bin/python -c 'from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset("data/external/yolo26x.pt", release="v8.4.0")'
```

The 113 MiB checkpoint stays outside Git. Its official source is
[Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt).
The model is loaded once and reused across windows. For a lighter option, pass
`--model data/external/yolo26n.pt --confidence 0.25` using the included nano
checkpoint. A non-editable installation needs an absolute `--model` path because
weights are stored in the source checkout.

## Choose devices and run

```bash
.venv/bin/python interface/main.py --list-devices
.venv/bin/python interface/main.py --camera /dev/video4 --once
.venv/bin/python interface/main.py --camera /dev/video4 --location study-room
```

`--camera` accepts a camera index (`0`, `4`, etc.), a Linux `/dev/videoN` path,
or a stable `/dev/v4l/by-id/...` symlink. On Linux, auto-selection matches the
selected USB webcam to its microphone using their common USB device. It fails
with instructions if no unique match exists. Device lists can include extra
video nodes that provide metadata rather than pictures; choose a capture node.

To choose an input explicitly, use its ID or name from `--list-devices`:

```bash
.venv/bin/python interface/main.py --camera 4 --audio-device 6
.venv/bin/python interface/main.py --camera 0 --audio-device default
```

On platforms without Linux device discovery, use the OpenCV camera index and
an explicit microphone ID/name (or `default`). The listing of camera names is
Linux-specific. IDs can change after devices are reconnected.

Optional settings: `--interval 60 --duration 10 --images 20 --sample-rate 48000
--confidence 0.15`. The audio rate defaults to the microphone's native rate.
`--once` exits after one analysis; otherwise stop with Ctrl+C. The webcam is
released between windows. Missed windows are skipped if analysis takes longer
than the interval, so inference time does not accumulate as scheduling drift.
Capture failures produce an error and exit instead of reporting false zeros.

Each completed window prints a JSON object with:

- `count`: integer person count, `mean_count`: unrounded mean, and
  `per_image_counts`: every individual count.
- `average_confidence`: mean score across detected boxes, or null when no people
  were detected; `per_image_average_confidences` preserves every frame. Detection
  confidence is not a measured accuracy rate.
- `decibels` (raw dBFS) and `rms`: full-recording digital level; `noise_level` and
  `noise_unit` (`relative`): category-aligned website index; `noise_band` and
  `noise_calibration`: category and profile ID.
- `image_count`, `audio_duration_seconds`, and `sample_rate`: capture details.
- `location` and `measured_at`: label and UTC recording-start time.
- `audio_file` and `image_files`: absolute paths to the retained capture.
- In upload mode: `room_id`, `device_id`, `upload_status`, and `uploaded_readings`.

The API stores integer counts and nullable `average_confidence` in readings,
including latest/history responses. Confidence is the mean person-box score,
not count accuracy or microphone confidence. No detections produce null. Existing
community trust weights are unchanged; the field is exposed for a later policy.

## Standalone capture for your own main

```python
from src.get_data import get_data

# Call once per minute from your own scheduler. Blocks for about 10 seconds.
audio_path, image_paths = get_data(camera=4)
# audio_path: Path to audio.wav; image_paths: list of 20 Paths to PNG images.
# Run your own analysis here. The WAV contains its sample rate.
```

This helper opens and closes the selected devices for each call. It performs
capture and file output only. Use `audio_device=6` (or `"default"`) to select a
microphone explicitly. The default directory is relative to the project, so it
works regardless of your current working directory; override with
`temp_dir="/another/directory"` if needed.

Every call creates a unique directory so it cannot overwrite a sample still
being processed. After processing, delete `audio_path.parent` and its contents
to reclaim space. Cleanup is the caller's responsibility; files do not expire
automatically. The lower-level `capture_sample()` still returns in-memory arrays.

## Verification

Hardware-free regression tests:

```bash
.venv/bin/python -m unittest discover -s testing -p 'test_analysis.py' -v
```

The analysis class also accepts an audio file and a list of image paths for
offline testing:

```python
from pathlib import Path
from src.process_data import ImageProcessor

data = Path("data/testing")
result = ImageProcessor().process_data(
    data / "file_example_WAV_1MG.wav",
    [data / "room.jpg", data / "room.png"],
    "test-room",
)
print(result)
```

Upstream standalone capture documentation is in `docs/capture.md`. The upstream
`Processor` single-image result schema remains available for compatibility;
`ImageProcessor` supports multi-frame sampling. `src/send_data.py` is a legacy
direct-Postgres writer for a different schema; the runner uses `src/upload.py`
and does not call it.

## Generated-image counting challenge

`data/testing/synthetic-2026-11-07/` contains seven generated scenes, frozen
pixel-reviewed occupant counts and generation prompts. Evaluation reports are
generated locally and excluded from Git.
These are synthetic challenge cases, not measured real-camera accuracy.
The frozen evaluator uses the nano checkpoint and its original 0.25 confidence threshold, at native input
size and after downsampling to 640×480, without any camera, microphone or uploads:

```bash
.venv/bin/python testing/evaluate_synthetic.py
```

Each run creates a fresh timestamped results directory with `report.html`,
`report.md`, `contact-sheet.png`, boxed predictions, model/source/image hashes,
exact dependency versions, and count-error metrics. The saved test images and
predictions are static files. The evaluator
uses silent fixture audio only to call the production analysis API, so it
establishes no microphone accuracy. It does not tune the model or threshold.

For the complete pulled regression suite, including optional legacy sender
tests that mock their database connection:

```bash
uv pip install --python .venv/bin/python -e '.[dev,legacy-send]'
.venv/bin/python -m pytest -q testing
```

Generate and verify the reproducible audio fixtures:

```sh
.venv/bin/python testing/generate_noise.py
```

From the adjacent app checkout, `npm run test:sensor` exercises real YOLO,
generated audio, authenticated HTTP, the configured database, confidence
readback and frontend categories, rolling back every temporary database row.
