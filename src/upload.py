"""Room-bound HTTPS uploads and a durable, metadata-only retry queue."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from http.client import HTTPException
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request


class UploadError(RuntimeError):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class DeviceConfig:
    api_url: str
    room_id: str
    device_id: str
    token: str = field(repr=False)

    def __post_init__(self):
        url = urllib.parse.urlsplit(self.api_url)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
                or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("API_URL must be an HTTP(S) origin without paths or credentials")
        if url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Device uploads require HTTPS except on localhost")
        try:
            port = url.port
        except ValueError:
            raise ValueError("API_URL has an invalid port") from None
        if port is not None and port == 0:
            raise ValueError("API_URL has an invalid port")
        for value in (self.room_id, self.device_id):
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
                raise ValueError("ROOM_ID and DEVICE_ID must be valid registered IDs")
        if not re.fullmatch(r"[0-9a-f]{64}", self.token):
            raise ValueError("DEVICE_TOKEN must be a 64-character hexadecimal device token")

    @classmethod
    def load(cls, path, env=None):
        values = {}
        if Path(path).is_file():
            for line in Path(path).read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip()
        env = os.environ if env is None else env
        for key in ("API_URL", "ROOM_ID", "DEVICE_ID", "DEVICE_TOKEN"):
            if key in env:
                values[key] = env[key]
        try:
            return cls(*(values[key] for key in ("API_URL", "ROOM_ID", "DEVICE_ID", "DEVICE_TOKEN")))
        except KeyError:
            raise ValueError("Upload requires API_URL, ROOM_ID, DEVICE_ID and DEVICE_TOKEN") from None


def reading_payload(result, room_id, measured_at):
    count = result["count"]
    confidence = result["average_confidence"]
    level = result["noise_level"]
    if type(count) is not int or not 0 <= count <= 2147483647:
        raise ValueError("Person count must be a nonnegative database integer")
    if confidence is not None and (type(confidence) not in (float, int) or not 0 <= confidence <= 1):
        raise ValueError("Detection confidence must be null or between 0 and 1")
    if result["noise_unit"] != "relative" or type(level) not in (int, float) or not 0 <= level <= 1:
        raise ValueError("Expected a calibrated relative noise level between 0 and 1")
    date = datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
    if date.utcoffset() is None:
        raise ValueError("Measurement time must have a timezone")
    return {"room_id": room_id, "people_count": count, "average_confidence": confidence,
            "noise_level": level, "noise_unit": "relative",
            "measured_at": date.astimezone(timezone.utc).isoformat(timespec="milliseconds")}


class ReadingUploader:
    def __init__(self, config, outbox):
        self.config = config
        path = Path(outbox)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS pending (device TEXT, room TEXT, measured TEXT, payload TEXT, PRIMARY KEY(device, measured))")
        self.db.commit()
        self.opener = urllib.request.build_opener(NoRedirects())

    def close(self):
        self.db.close()

    def enqueue(self, payload):
        if payload["room_id"] != self.config.room_id:
            raise ValueError("Queued reading belongs to another room")
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        with self.db:
            row = self.db.execute("SELECT payload FROM pending WHERE device=? AND measured=?",
                                  (self.config.device_id, payload["measured_at"])).fetchone()
            if row and row[0] != encoded:
                raise ValueError("A different reading is already queued for this timestamp")
            self.db.execute("INSERT OR IGNORE INTO pending VALUES (?, ?, ?, ?)",
                            (self.config.device_id, self.config.room_id, payload["measured_at"], encoded))

    def send(self, payload):
        request = urllib.request.Request(
            self.config.api_url.rstrip("/") + "/api/readings", method="POST",
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.config.token})
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=5) as response:
                    saved = json.load(response)["reading"]
                if saved["device_id"] != self.config.device_id or saved["room_id"] != self.config.room_id:
                    raise UploadError("API acknowledgement did not match configured device/room")
                return saved
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if status < 500 or attempt == 2:
                    raise UploadError(f"Upload rejected (HTTP {status}); reading retained in outbox") from None
            except (urllib.error.URLError, OSError, HTTPException):
                if attempt == 2:
                    raise UploadError("API unreachable; reading retained in outbox") from None
            except (ValueError, KeyError, TypeError):
                raise UploadError("Invalid API acknowledgement; reading retained in outbox") from None
            time.sleep(0.5 * 2 ** attempt)

    def flush(self, limit=5):
        # Current observations first so an outage backlog cannot hide live data.
        rows = self.db.execute("SELECT measured, payload FROM pending WHERE device=? AND room=? ORDER BY measured DESC LIMIT ?",
                               (self.config.device_id, self.config.room_id, limit)).fetchall()
        for measured, payload in rows:
            self.send(json.loads(payload))
            with self.db:
                self.db.execute("DELETE FROM pending WHERE device=? AND measured=?", (self.config.device_id, measured))
        return len(rows)
