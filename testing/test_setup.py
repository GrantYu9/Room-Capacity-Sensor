"""Quick setup without cameras, database mutations, real tokens or capture."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import threading
from unittest.mock import Mock, patch

import pytest

from interface import setup


TOKEN = "a" * 64
ROOM = {"room_id": "room-123", "name": "North Reading Room", "location": {"building": "Library"}}


def profile(path, room="room-123", api="https://example.com"):
    path.write_text(f"API_URL={api}\nROOM_ID={room}\nDEVICE_ID=test-sensor\nDEVICE_TOKEN={TOKEN}\n")
    path.chmod(0o600)
    return path, setup.load_profile(path)


@pytest.mark.parametrize("caps,node_caps,expected", [
    (0x80000001, 0x00800000, False),  # device metadata node, despite global capture capability
    (0x80000001, 1, True),
    (0x80000001, 0x1000, True),
    (1, 0, True),
])
def test_queries_node_specific_capture_capability_without_streaming(caps, node_caps, expected):
    def query(fd, request, buffer):
        assert (fd, request) == (17, 0x80685600)
        struct.pack_into("=II", buffer, 84, caps, node_caps)

    with patch.object(setup.os, "open", return_value=17) as opened, \
         patch.object(setup.os, "close") as closed, patch("fcntl.ioctl", side_effect=query):
        assert setup.capture_capable("/dev/video4") == expected
    opened.assert_called_once_with("/dev/video4", os.O_RDONLY | os.O_NONBLOCK)
    closed.assert_called_once_with(17)


def test_numbered_fuzzy_search_accepts_typos_word_order_and_accents(monkeypatch):
    room = {"room_id": "library-123"}
    options = [("library-123 · North Reading Room · Bibliothèque", room), ("room-9 · Cafeteria", {})]
    answers = iter(["readng north", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert setup.choose("Room", options, fuzzy=True, use_fzf=False) is room
    assert setup.fuzzy_score("bibliotheque", options[0][0]) > 0
    assert setup.fuzzy_score("library-123", options[0][0]) > setup.fuzzy_score("library-123", options[1][0])
    assert setup.fuzzy_score("zzzzzz", options[0][0]) == 0
    assert "\x1b" not in setup.display("name\x1b[31m\nnext")


def test_fzf_returns_original_value_and_cancel_stops_setup():
    room = dict(ROOM)
    with patch.object(setup.shutil, "which", return_value="/usr/bin/fzf"), \
         patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "1\troom label\n")) as run:
        assert setup.choose("Room", [("first", {}), ("second", room)]) is room
        assert run.call_args.kwargs["env"]["FZF_DEFAULT_OPTS"] == ""
    with patch.object(setup.shutil, "which", return_value="/usr/bin/fzf"), \
         patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 130, "")), pytest.raises(KeyboardInterrupt):
        setup.choose("Room", [("first", room)])


@contextmanager
def rooms_api(body):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/rooms"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_website_rooms_keep_exact_ids_and_malformed_catalogs_fail():
    with rooms_api({"rooms": [ROOM]}) as url:
        assert setup.fetch_rooms(url) == [ROOM]
    for body in [{"rooms": []}, {"rooms": [ROOM, ROOM]}, {"rooms": [{**ROOM, "room_id": "../bad"}]},
                 {"rooms": [{**ROOM, "location": "bad"}]}]:
        with rooms_api(body) as url, pytest.raises(setup.SetupError):
            setup.fetch_rooms(url)


def test_existing_profile_reused_only_for_exact_room_and_origin(tmp_path, monkeypatch):
    selected = profile(tmp_path / ".env.sensor")
    monkeypatch.setenv("ROOM_ID", "another-room")
    monkeypatch.setenv("DEVICE_TOKEN", "b" * 64)
    assert setup.load_profile(selected[0]).room_id == ROOM["room_id"]
    with patch("builtins.input", side_effect=AssertionError("Matching profiles need no provisioning")):
        assert setup.select_profile("https://example.com", ROOM, [selected], tmp_path) == selected
    wrong = profile(tmp_path / ".env.wrong", room="other-room")
    with patch("builtins.input", return_value=str(wrong[0])), pytest.raises(setup.SetupError, match="another room"):
        setup.select_profile("https://example.com", ROOM, [wrong], tmp_path)


def test_registration_preserves_existing_credentials_and_does_not_pass_tokens_as_args(tmp_path):
    backend = tmp_path / "backend"
    (backend / "scripts").mkdir(parents=True)
    (backend / "scripts/add-device.js").touch()
    output, old = profile(tmp_path / ".env.old")
    with patch.object(setup.subprocess, "run") as run, pytest.raises(setup.SetupError, match="already exists"):
        setup.register_device(backend, "https://example.com", ROOM, "test-sensor", output)
    run.assert_not_called()
    assert setup.load_profile(output) == old
    output = tmp_path / ".env.new"

    def registered(command, **kwargs):
        assert TOKEN not in " ".join(command)
        assert command[command.index("--room") + 1] == ROOM["room_id"]
        profile(output)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    with patch.object(setup.shutil, "which", return_value="node"), \
         patch.object(setup.subprocess, "run", side_effect=registered):
        saved, config = setup.register_device(backend, "https://example.com", ROOM, "test-sensor", output)
    assert saved == output
    assert config.room_id == ROOM["room_id"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_launcher_quotes_paths_forwards_args_and_clears_shell_credential_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    runner = tmp_path / "runner with spaces.py"
    runner.write_text("import os,sys,json; print(json.dumps({'args':sys.argv[1:], 'overrides':[k for k in ['API_URL','ROOM_ID','DEVICE_ID','DEVICE_TOKEN'] if k in os.environ]}))")
    launch = tmp_path / "run-sensor.local.sh"
    tricky = "private profile $(not-a-command) 'quote'"
    setup.save_launcher(launch, [sys.executable, str(runner), "--env", tricky])
    result = subprocess.run([str(launch), "--once"], check=True, capture_output=True, text=True,
                            env={**os.environ, **{key: TOKEN for key in setup.ENV_KEYS}})
    assert json.loads(result.stdout) == {"args": ["--env", tricky, "--once"], "overrides": []}
    assert TOKEN not in launch.read_text()
    assert stat.S_IMODE(launch.stat().st_mode) == 0o700
    # Rerunning setup can update its launcher but cannot replace unrelated files.
    setup.save_launcher(launch, [sys.executable, str(runner)])
    launch.write_text("user-owned script")
    with pytest.raises(setup.SetupError, match="unrelated"):
        setup.save_launcher(launch, ["true"])
    assert launch.read_text() == "user-owned script"


def test_configure_only_saves_complete_launcher_without_starting_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    selected = profile(tmp_path / ".env.laptop")
    monkeypatch.setattr(setup, "profiles", lambda: [selected])
    monkeypatch.setattr(setup, "fetch_rooms", lambda url: [ROOM])
    monkeypatch.setattr(setup, "discover_cameras", lambda: [{"device": "/dev/video4", "name": "Brio"}])
    monkeypatch.setattr(setup, "choose", lambda title, options, **kwargs: options[0][1])
    with patch("src.get_data.resolve_microphone", return_value=6), \
         patch.object(setup.subprocess, "call") as start, \
         patch("builtins.input", side_effect=AssertionError("configure-only must not ask to start")):
        assert setup.main(["--configure-only"]) == 0
    start.assert_not_called()
    launcher = (tmp_path / "run-sensor.local.sh").read_text()
    assert "--camera /dev/video4 --audio-device 6" in launcher
    assert str(selected[0]) in launcher
    assert TOKEN not in launcher


def test_cancellation_does_not_write_a_launcher_or_register_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(setup, "profiles", lambda: [])
    monkeypatch.setattr(setup, "fetch_rooms", lambda url: [ROOM])
    monkeypatch.setattr(setup, "discover_cameras", lambda: [{"device": "/dev/video4", "name": "Brio"}])
    monkeypatch.setattr(setup, "choose", Mock(side_effect=KeyboardInterrupt))
    assert setup.main(["--configure-only"]) == 130
    assert not (tmp_path / "run-sensor.local.sh").exists()
