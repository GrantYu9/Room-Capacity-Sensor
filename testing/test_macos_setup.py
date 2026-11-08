"""Simulate macOS discovery/setup on CI without opening hardware or macOS APIs."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from packaging.requirements import Requirement
import pytest

from interface import setup
from src import get_data


@pytest.fixture
def mac_platform(monkeypatch):
    # Patch only these modules, not Python's global platform identity.
    monkeypatch.setattr(setup, "sys", SimpleNamespace(platform="darwin", executable=sys.executable, stderr=sys.stderr))
    monkeypatch.setattr(get_data, "sys", SimpleNamespace(platform="darwin"))


def native_device(uid, name):
    return SimpleNamespace(uniqueID=lambda: uid, localizedName=lambda: name)


def test_macos_names_and_indices_follow_opencv_video_plus_muxed_id_order(mac_platform, monkeypatch):
    av = SimpleNamespace(AVMediaTypeVideo="video", AVMediaTypeMuxed="muxed",
                         AVCaptureDevice=SimpleNamespace(devicesWithMediaType_=Mock(side_effect=[
                             [native_device("z-usb", "USB Webcam"), native_device("a-built-in", "FaceTime HD Camera")],
                             [native_device("m-muxed", "Capture Device")],
                         ])))
    monkeypatch.setitem(sys.modules, "AVFoundation", av)
    with patch.object(get_data.cv2, "VideoCapture") as capture:
        cameras = setup.discover_cameras()
    assert cameras == [
        {"device": 0, "name": "FaceTime HD Camera"},
        {"device": 1, "name": "Capture Device"},
        {"device": 2, "name": "USB Webcam"},
    ]
    assert av.AVCaptureDevice.devicesWithMediaType_.call_args_list[0].args == ("video",)
    assert av.AVCaptureDevice.devicesWithMediaType_.call_args_list[1].args == ("muxed",)
    capture.assert_not_called()


def test_missing_macos_bindings_show_dependency_install_command(mac_platform, monkeypatch):
    monkeypatch.setitem(sys.modules, "AVFoundation", None)
    with pytest.raises(setup.SetupError, match="pip install -e"):
        setup.discover_cameras()


def test_macos_capture_uses_same_avfoundation_index_backend(mac_platform):
    with patch.object(get_data.cv2, "VideoCapture") as capture:
        with get_data.Photographer(2):
            pass
    capture.assert_called_once_with(2, get_data.cv2.CAP_AVFOUNDATION)
    capture.return_value.read.assert_not_called()
    capture.return_value.release.assert_called_once()


def test_macos_open_failure_explains_camera_permission(mac_platform):
    with patch.object(get_data.cv2, "VideoCapture") as capture:
        capture.return_value.isOpened.return_value = False
        with pytest.raises(RuntimeError, match="Allow Camera access"):
            get_data.Photographer(0)
    capture.return_value.release.assert_called_once()


def test_macos_setup_chooses_camera_and_separate_microphone_and_saves_launcher(mac_platform, tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    profile = tmp_path / ".env.mac"
    profile.write_text("API_URL=https://example.com\nROOM_ID=room-1\nDEVICE_ID=mac-sensor\nDEVICE_TOKEN=" + "a" * 64 + "\n")
    profile.chmod(0o600)
    room = {"room_id": "room-1", "name": "Reading Room", "location": {}}
    monkeypatch.setattr(setup, "fetch_rooms", lambda url: [room])
    monkeypatch.setattr(setup, "discover_cameras", lambda: [{"device": 0, "name": "FaceTime HD"}, {"device": 1, "name": "USB Webcam"}])
    selections = []

    def select(title, options, **kwargs):
        selections.append(title)
        return options[1 if title == "Webcam" else 0][1]

    monkeypatch.setattr(setup, "choose", select)
    with patch.object(get_data, "resolve_microphone", side_effect=AssertionError("Linux matching must not run on Mac")), \
         patch.object(get_data, "list_microphones", return_value=[{"device": 7, "name": "MacBook Microphone"}]), \
         patch.object(setup.subprocess, "call") as start:
        assert setup.main(["--env", str(profile), "--configure-only"]) == 0
    assert selections == ["Webcam", "Microphone", "Room"]
    launcher = (tmp_path / "run-sensor.local.sh").read_text()
    assert "--camera 1 --audio-device 7" in launcher
    assert str(profile) in launcher
    assert "a" * 64 not in launcher
    start.assert_not_called()


def test_macos_microphone_error_explains_permissions(mac_platform):
    with patch.object(get_data, "list_microphones", side_effect=RuntimeError("private diagnostics")), \
         pytest.raises(setup.SetupError, match="allow Microphone access") as error:
        setup.select_microphone("0", use_fzf=False)
    assert "private diagnostics" not in str(error.value)


def test_macos_manual_fallback_rejects_linux_device_paths(mac_platform, tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(setup, "profiles", lambda: [])
    monkeypatch.setattr(setup, "fetch_rooms", lambda url: [{"room_id": "room-1"}])
    monkeypatch.setattr(setup, "discover_cameras", lambda: [])
    monkeypatch.setattr("builtins.input", lambda prompt: "/dev/video4")
    assert setup.main(["--configure-only"]) == 1
    assert not (tmp_path / "run-sensor.local.sh").exists()


def test_native_dependency_is_macos_only():
    import tomllib
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    dependency = next(Requirement(spec) for spec in project["project"]["dependencies"] if spec.startswith("pyobjc-framework-AVFoundation"))
    assert dependency.marker.evaluate({"sys_platform": "darwin"})
    assert not dependency.marker.evaluate({"sys_platform": "linux"})
    assert not dependency.marker.evaluate({"sys_platform": "win32"})


def test_setup_finds_renamed_and_legacy_backend_checkouts(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path / "Room-Capacity-Sensor")
    legacy = tmp_path / "Room-Capacity-App"
    modern = tmp_path / "Room-Capacity-Website"
    assert setup.default_backend() == modern
    (legacy / "scripts").mkdir(parents=True)
    (legacy / "scripts/add-device.js").touch()
    assert setup.default_backend() == legacy
    (modern / "scripts").mkdir(parents=True)
    (modern / "scripts/add-device.js").touch()
    assert setup.default_backend() == modern
