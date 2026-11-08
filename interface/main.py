"""Run local analysis; one JSON result per sampling window on stdout."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

# Support both `python interface/main.py` and `python -m interface.main`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.get_data import get_data, list_cameras, list_microphones
from src.process_data import ImageProcessor
from src.noise import NoiseCalibration, DEFAULT_PROFILE
from src.upload import DeviceConfig, ReadingUploader, UploadError, reading_payload


def device_argument(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="List cameras and microphone IDs, then exit")
    parser.add_argument("--camera", type=device_argument, default=0, help="OpenCV camera index or /dev/video path")
    parser.add_argument("--audio-device", type=device_argument, help="Microphone ID/name, or 'default'; otherwise match USB webcam")
    parser.add_argument("--interval", type=positive_float, default=60, help="Seconds between window starts (default: 60)")
    parser.add_argument("--duration", type=positive_float, default=10, help="Recording duration in seconds (default: 10)")
    parser.add_argument("--images", type=positive_int, default=20, help="Images spread across each recording (default: 20)")
    parser.add_argument("--sample-rate", type=positive_int, help="Audio Hz; defaults to the microphone's native rate")
    parser.add_argument("--location", default="", help="Location label included in the result")
    parser.add_argument("--model", type=Path, help="Local YOLO detection weights")
    parser.add_argument("--confidence", type=positive_float, default=0.15)
    parser.add_argument("--once", action="store_true", help="Analyze one window, then exit")
    parser.add_argument("--upload", action="store_true", help="Write each completed window to the authenticated readings API")
    parser.add_argument("--env", type=Path, default=Path(".env.sensor"), help="Private device configuration file")
    parser.add_argument("--outbox", type=Path, default=Path(__file__).resolve().parents[1] / "data/outbox/readings.sqlite3")
    parser.add_argument("--noise-calibration", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args(argv)
    if args.duration > args.interval:
        parser.error("--duration must not exceed --interval")
    if args.confidence > 1:
        parser.error("--confidence must not exceed 1")
    return args


def run(args):
    if args.list_devices:
        print(json.dumps({"cameras": list_cameras(), "microphones": list_microphones()}, indent=2))
        return
    config = DeviceConfig.load(args.env) if args.upload else None
    uploader = ReadingUploader(config, args.outbox) if config else None
    try:
        run_windows(args, uploader)
    finally:
        if uploader:
            uploader.close()


def run_windows(args, uploader=None):
    processor = ImageProcessor(args.model, args.confidence, NoiseCalibration.load(args.noise_calibration))
    print(f"Camera: {args.camera}; microphone: {args.audio_device if args.audio_device is not None else 'webcam auto'}; "
          f"{args.duration}s / {args.images} images every {args.interval}s", file=sys.stderr, flush=True)
    next_window = time.monotonic()
    while True:
        time.sleep(max(0, next_window - time.monotonic()))
        measured_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        audio, images, sample_rate = get_data(
            args.camera, args.audio_device, duration=args.duration,
            image_count=args.images, sample_rate=args.sample_rate,
        )
        result = processor.process_data(audio, images, args.location, sample_rate=sample_rate)
        result["measured_at"] = measured_at
        if uploader:
            result["room_id"] = uploader.config.room_id
            result["device_id"] = uploader.config.device_id
            uploader.enqueue(reading_payload(result, uploader.config.room_id, measured_at))
            try:
                result["uploaded_readings"] = uploader.flush()
                result["upload_status"] = "saved"
            except UploadError as error:
                result["upload_status"] = "queued"
                print(str(error), file=sys.stderr, flush=True)
                if args.once:
                    print(json.dumps(result, allow_nan=False), flush=True)
                    raise
        print(json.dumps(result, allow_nan=False), flush=True)
        if args.once:
            return
        next_window += args.interval
        # Keep minute boundaries; skip missed windows if inference runs long.
        now = time.monotonic()
        if next_window < now:
            skipped = math.floor((now - next_window) / args.interval) + 1
            print(f"Analysis overran the interval; skipping {skipped} window(s)", file=sys.stderr)
            next_window += skipped * args.interval


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 0
    except Exception as error:
        print(f"Sensor error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
