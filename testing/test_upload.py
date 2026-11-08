"""Exercise real HTTP, durable retries, mapping, and the scheduled pipeline."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch, Mock

import pytest

from interface import main as app
from src.noise import NoiseCalibration, audio_level, fit_profile
from src.upload import DeviceConfig, ReadingUploader, UploadError, reading_payload


TOKEN = "a" * 64
RESULT = {"count": 2, "average_confidence": 0.8, "noise_level": 0.4, "noise_unit": "relative"}
TIME = "2026-11-08T12:00:00.123456+00:00"


@contextmanager
def api(statuses=None, device="pi-1", redirect=None):
    received = []
    statuses = list(statuses or [200])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((body, self.headers.get("Authorization")))
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            self.send_response(status)
            if redirect:
                self.send_header("Location", redirect)
            self.end_headers()
            self.wfile.write(json.dumps({"reading": {**body, "device_id": device}}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_upload_retries_identical_payload_and_survives_queue_reopen(tmp_path):
    with api([503, 200]) as (url, received), patch("src.upload.time.sleep"):
        config = DeviceConfig(url, "room-1", "pi-1", TOKEN)
        first = ReadingUploader(config, tmp_path / "outbox.sqlite3")
        payload = reading_payload(RESULT, "room-1", TIME)
        assert payload["measured_at"] == "2026-11-08T12:00:00.123+00:00"
        first.enqueue(payload)
        first.close()
        reopened = ReadingUploader(config, tmp_path / "outbox.sqlite3")
        try:
            assert reopened.flush() == 1
            assert reopened.flush() == 0
            assert len(received) == 2
            assert received[0] == received[1]
            assert received[0][1] == "Bearer " + TOKEN
            assert received[0][0]["average_confidence"] == 0.8
            assert "device_id" not in received[0][0]  # Server resolves identity.
        finally:
            reopened.close()


@pytest.mark.parametrize("status", [401, 403, 409, 503])
def test_rejected_reading_is_retained(tmp_path, status):
    with api([status]) as (url, received), patch("src.upload.time.sleep"):
        uploader = ReadingUploader(DeviceConfig(url, "room-1", "pi-1", TOKEN), tmp_path / "queue.sqlite3")
        try:
            uploader.enqueue(reading_payload(RESULT, "room-1", TIME))
            with pytest.raises(UploadError, match=f"HTTP {status}"):
                uploader.flush()
            assert uploader.db.execute("SELECT count(*) FROM pending").fetchone()[0] == 1
            assert len(received) == (3 if status == 503 else 1)
        finally:
            uploader.close()


def test_redirects_do_not_forward_credentials_and_wrong_device_ack_is_not_deleted(tmp_path):
    with api() as (destination, intercepted), api([307], redirect=destination) as (origin, received):
        uploader = ReadingUploader(DeviceConfig(origin, "room-1", "pi-1", TOKEN), tmp_path / "redirect.sqlite3")
        try:
            uploader.enqueue(reading_payload(RESULT, "room-1", TIME))
            with pytest.raises(UploadError, match="HTTP 307"):
                uploader.flush()
            assert not intercepted
            assert len(received) == 1
        finally:
            uploader.close()
    with api(device="wrong-device") as (origin, _):
        uploader = ReadingUploader(DeviceConfig(origin, "room-1", "pi-1", TOKEN), tmp_path / "ack.sqlite3")
        try:
            uploader.enqueue(reading_payload(RESULT, "room-1", TIME))
            with pytest.raises(UploadError, match="did not match"):
                uploader.flush()
            assert uploader.db.execute("SELECT count(*) FROM pending").fetchone()[0] == 1
        finally:
            uploader.close()


def test_mapping_and_config_errors_fail_before_capture(tmp_path):
    for url in ["http://example.com", "https://example.com/path", "https://user:pass@example.com",
                "https://example.com:invalid", "https://example.com:0"]:
        with pytest.raises(ValueError):
            DeviceConfig(url, "room-1", "pi-1", TOKEN)
    with patch.object(app, "get_data") as capture, redirect_stderr(io.StringIO()):
        assert app.main(["--upload", "--env", str(tmp_path / "missing.env")]) == 1
        capture.assert_not_called()
    config = DeviceConfig("https://example.com", "room-1", "pi-1", TOKEN)
    assert TOKEN not in repr(config)
    uploader = ReadingUploader(config, tmp_path / "queue.sqlite3")
    try:
        payload = reading_payload(RESULT, "room-1", TIME)
        uploader.enqueue(payload)
        with pytest.raises(ValueError):
            uploader.enqueue({**payload, "people_count": 5})
        with pytest.raises(ValueError):
            uploader.enqueue({**payload, "room_id": "room-2"})
    finally:
        uploader.close()


def test_payload_uses_utc_and_keeps_empty_detection_confidence_unknown():
    empty = {**RESULT, "count": 0, "average_confidence": None}
    result = reading_payload(empty, "room-1", "2026-11-08T04:00:00.123456-08:00")
    assert result["measured_at"] == "2026-11-08T12:00:00.123+00:00"
    assert result["people_count"] == 0
    assert result["average_confidence"] is None


def test_truncated_http_response_retries_and_retains_on_exhaustion(tmp_path):
    class BrokenResponse(io.BytesIO):
        def read(self, *args):
            raise IncompleteRead(b'{"reading":', 100)

    payload = reading_payload(RESULT, "room-1", TIME)
    saved = {**payload, "device_id": "pi-1"}
    uploader = ReadingUploader(DeviceConfig("https://example.com", "room-1", "pi-1", TOKEN), tmp_path / "queue.sqlite3")
    uploader.opener = Mock()
    try:
        uploader.enqueue(payload)
        uploader.opener.open.side_effect = [BrokenResponse(), io.BytesIO(json.dumps({"reading": saved}).encode())]
        with patch("src.upload.time.sleep"):
            assert uploader.flush() == 1
        assert uploader.opener.open.call_count == 2
        uploader.enqueue(payload)
        uploader.opener.open.side_effect = [BrokenResponse(), BrokenResponse(), BrokenResponse()]
        with patch("src.upload.time.sleep"), pytest.raises(UploadError, match="unreachable"):
            uploader.flush()
        assert uploader.db.execute("SELECT count(*) FROM pending").fetchone()[0] == 1
    finally:
        uploader.close()


def test_runner_posts_then_removes_media_and_keeps_confidence(tmp_path):
    directory = tmp_path / "capture-fixture"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.touch()
    with api() as (url, received):
        config = tmp_path / "device.env"
        config.write_text(f"API_URL={url}\nROOM_ID=room-1\nDEVICE_ID=pi-1\nDEVICE_TOKEN={TOKEN}\n")
        with patch.object(app, "get_data", return_value=(audio, [])), patch.object(app, "ImageProcessor") as processor, \
             redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            processor.return_value.process_data.return_value = dict(RESULT)
            assert app.main(["--once", "--upload", "--env", str(config), "--outbox", str(tmp_path / "queue.sqlite3")]) == 0
        result = json.loads(stdout.getvalue())
        assert result["upload_status"] == "saved"
        assert received[0][0]["people_count"] == 2
        assert received[0][0]["average_confidence"] == 0.8
        assert not directory.exists()


def test_recurring_runner_keeps_collecting_after_failed_upload(tmp_path):
    uploader = Mock()
    uploader.config = DeviceConfig("https://example.com", "room-1", "pi-1", TOKEN)
    uploader.flush.side_effect = [UploadError("offline; retained"), 1]
    with patch.object(app.DeviceConfig, "load", return_value=uploader.config), \
         patch.object(app, "ReadingUploader", return_value=uploader), \
         patch.object(app, "get_data", side_effect=[(Path("audio.wav"), []), (Path("audio.wav"), []), KeyboardInterrupt]), \
         patch.object(app, "ImageProcessor") as processor, patch.object(app.time, "sleep"), \
         redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
        processor.return_value.process_data.side_effect = [dict(RESULT), dict(RESULT)]
        assert app.main(["--upload"]) == 0
    windows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [window["upload_status"] for window in windows] == ["queued", "saved"]
    assert uploader.enqueue.call_count == 2
    uploader.close.assert_called_once()


def test_generated_audio_calibration_and_held_out_checks(tmp_path):
    from testing.generate_noise import generate
    profile, results = generate(tmp_path)
    assert len(results) == 9
    assert {row["expected_band"] for row in results} == {"quiet", "moderate", "lively"}
    shipped = NoiseCalibration.load()
    assert profile.quiet_max_dbfs == pytest.approx(shipped.quiet_max_dbfs, abs=1e-5)
    assert profile.moderate_max_dbfs == pytest.approx(shipped.moderate_max_dbfs, abs=1e-5)
    for row in results:
        assert shipped.classify(row["decibels"])["noise_band"] == row["expected_band"]
    for dbfs, expected, boundary in [(-47.501, "quiet", None), (-47.5, "moderate", 0.25),
                                     (-27.001, "moderate", None), (-27, "lively", 0.6)]:
        result = shipped.classify(dbfs)
        assert result["noise_band"] == expected
        if boundary is not None:
            assert result["noise_level"] == boundary
    assert shipped.classify(-120)["noise_level"] == 0
    manifest = tmp_path / "calibration.json"
    data = json.loads(manifest.read_text())
    # Overlapping real-room labels must be rejected, rather than silently fitted.
    data["samples"][0]["band"] = "lively"
    data["samples"].append({**data["samples"][1], "band": "quiet"})
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap"):
        fit_profile(manifest, "invalid", "fixture")
