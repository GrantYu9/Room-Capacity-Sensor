"""Pick a webcam and website room, then save a local sensor launcher."""

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.upload import DeviceConfig, NoRedirects


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = "https://sensoryatlas.vercel.app"
ENV_KEYS = ("API_URL", "ROOM_ID", "DEVICE_ID", "DEVICE_TOKEN")
ID = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")


class SetupError(Exception):
    pass


def display(value):
    # API/device names are text, never terminal control sequences or shell code.
    return " ".join("".join(c for c in str(value or "") if c.isprintable()).split())[:200]


def origin(value):
    return DeviceConfig(value, "setup", "setup", "0" * 64).api_url.rstrip("/")


def capture_capable(path):
    """Query Linux node capabilities without starting a video stream."""
    import fcntl

    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        capability = bytearray(104)  # struct v4l2_capability
        fcntl.ioctl(fd, 0x80685600, capability)  # VIDIOC_QUERYCAP
        capabilities, device_caps = struct.unpack_from("=II", capability, 84)
        node_caps = device_caps if capabilities & 0x80000000 else capabilities
        return bool(node_caps & 0x1001)  # VIDEO_CAPTURE / VIDEO_CAPTURE_MPLANE
    finally:
        os.close(fd)


def discover_cameras():
    if sys.platform == "darwin":
        from src.get_data import list_cameras
        try:
            return list_cameras()
        except RuntimeError as error:
            raise SetupError(str(error)) from None
    cameras = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        device = Path("/dev") / node.name
        try:
            if not capture_capable(device):
                continue
            name = (node / "name").read_text().strip()
        except OSError:
            continue
        links = sorted(p for p in Path("/dev/v4l/by-id").glob("*") if p.resolve() == device.resolve())
        cameras.append({"device": str(links[0] if links else device), "name": name})
    return cameras


def normalize(value):
    plain = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(re.findall(r"[a-z0-9]+", "".join(c for c in plain if not unicodedata.combining(c))))


def fuzzy_score(query, label):
    query, label = normalize(query), normalize(label)
    if not query:
        return 1
    if query in label:
        return 2 + len(query) / max(len(label), 1)
    words = label.split()
    scores = [max((difflib.SequenceMatcher(None, part, word).ratio() for word in words), default=0)
              for part in query.split()]
    return sum(scores) / len(scores) if min(scores) >= 0.55 else 0


def choose(title, options, *, fuzzy=False, use_fzf=True):
    """Return an original value; picker labels are never treated as commands."""
    if not options:
        raise SetupError(f"No options available: {title}")
    if use_fzf and shutil.which("fzf"):
        lines = [f"{i}\t{display(label)}" for i, (label, _) in enumerate(options)]
        env = {**os.environ, "FZF_DEFAULT_OPTS": "", "FZF_DEFAULT_OPTS_FILE": ""}
        result = subprocess.run(
            ["fzf", "--height=60%", "--layout=reverse", "--border", "--no-multi",
             "--delimiter=\t", "--with-nth=2..", f"--prompt={title}> ",
             "--header=Type to search; Enter selects; Esc cancels"],
            input="\n".join(lines) + "\n", text=True, stdout=subprocess.PIPE, env=env,
        )
        if result.returncode in (1, 130):
            raise KeyboardInterrupt
        if result.returncode != 0:
            raise SetupError("fzf could not start. Retry with --no-fzf.")
        try:
            index = int(result.stdout.split("\t", 1)[0])
            if not 0 <= index < len(options):
                raise ValueError
            return options[index][1]
        except (ValueError, IndexError):
            raise SetupError("The picker returned an invalid selection.") from None
    query = ""
    while True:
        print(f"\n{title}")
        if fuzzy:
            query = input("Search by room ID, name or building (Enter lists all; q cancels): ").strip()
            if query.lower() == "q":
                raise KeyboardInterrupt
        ranked = sorted(((fuzzy_score(query, label), label, value) for label, value in options),
                        key=lambda item: -item[0]) if fuzzy else [(1, label, value) for label, value in options]
        matches = [(label, value) for score, label, value in ranked if score > 0][:20]
        if not matches:
            print("No matches; try a shorter name or room ID.")
            continue
        for i, (label, _) in enumerate(matches, 1):
            print(f"  {i}. {display(label)}")
        answer = input("Choose a number (Enter searches again; q cancels): ").strip()
        if answer.lower() == "q":
            raise KeyboardInterrupt
        if answer.isdecimal() and 1 <= int(answer) <= len(matches):
            return matches[int(answer) - 1][1]
        if answer:
            print("Choose one of the displayed numbers.")


def fetch_rooms(api_url):
    try:
        request = urllib.request.Request(origin(api_url) + "/api/rooms", headers={"Accept": "application/json"})
        with urllib.request.build_opener(NoRedirects()).open(request, timeout=15) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError
        rooms = json.loads(data)["rooms"]
        if not isinstance(rooms, list) or not rooms:
            raise ValueError
        if any(not isinstance(room, dict) or not isinstance(room.get("location") or {}, dict) for room in rooms):
            raise ValueError
        ids = [room["room_id"] for room in rooms]
        if any(not isinstance(value, str) or not ID.fullmatch(value) for value in ids) or len(set(ids)) != len(ids):
            raise ValueError
        return rooms
    except (OSError, ValueError, KeyError, TypeError):
        raise SetupError("Could not load website rooms. Check --api-url and the connection, then retry.") from None


def room_label(room):
    location = room.get("location") or {}
    return " · ".join(display(value) for value in
                      [room["room_id"], room.get("name"), location.get("building"),
                       location.get("room_number"), location.get("address")] if value)


def load_profile(path):
    try:
        return DeviceConfig.load(path, env={})
    except (OSError, ValueError):
        raise SetupError("Could not load that device profile. Use a private file generated by device:add.") from None


def profiles():
    found = []
    for path in sorted(ROOT.glob(".env.*")):
        if path.name == ".env.example" or not path.is_file() or path.is_symlink():
            continue
        try:
            found.append((path, load_profile(path)))
        except SetupError:
            pass
    return found


def register_device(backend, api_url, room, device_id, output):
    if not ID.fullmatch(device_id) or device_id.startswith("demo-"):
        raise SetupError("Device ID must be 1–64 letters, numbers, underscores or hyphens; avoid demo-.")
    if output.exists() or output.is_symlink():
        raise SetupError("That credential file already exists; existing credentials were preserved.")
    script = backend / "scripts/add-device.js"
    if not script.is_file() or not shutil.which("node"):
        raise SetupError("The trusted app backend and Node.js are required for registration.")
    # The backend reads its own DATABASE_URL. Never bring it onto the sensor or
    # display child output, which may contain environment-specific diagnostics.
    result = subprocess.run(
        ["node", str(script), "--room", room["room_id"], "--name", room.get("name") or room["room_id"],
         "--device", device_id, "--api-url", api_url, "--output", str(output)],
        cwd=backend, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode:
        raise SetupError("Registration failed. Check backend configuration, migrations, and that the device ID is new. Existing profiles were preserved.")
    config = load_profile(output)
    if (config.room_id, config.device_id, origin(config.api_url)) != (room["room_id"], device_id, api_url):
        raise SetupError("The registered profile did not match the selected room/device/API; sensor startup stopped.")
    return output, config


def select_profile(api_url, room, available, backend, *, use_fzf=True):
    matching = [(path, config) for path, config in available
                if config.room_id == room["room_id"] and origin(config.api_url) == api_url]
    if matching:
        if len(matching) == 1:
            selected = matching[0]
        else:
            selected = choose("Device profile", [(f"{config.device_id} · {path.name}", (path, config))
                                                  for path, config in matching], use_fzf=use_fzf)
        print(f"Reusing registered device {display(selected[1].device_id)} from {selected[0].name}.")
        return selected
    print("\nThis room needs its own registered device token; an existing token cannot be reassigned here.")
    can_register = (backend / "scripts/add-device.js").is_file()
    if can_register:
        print(f"Trusted backend available: {backend}")
        answer = input("Register a NEW device for this room using that backend? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            host = re.sub(r"[^a-zA-Z0-9_-]", "-", socket.gethostname())[:32] or "sensor"
            default = f"{host}-{secrets.token_hex(3)}"
            device = input(f"New device ID [{default}]: ").strip() or default
            output = ROOT / f".env.sensor-{device}"
            selected = register_device(backend, api_url, room, device, output)
            print(f"Registered {display(device)}. Private credentials saved to {output.name}.")
            return selected
    path = input("Path to an already registered private device profile (Enter cancels): ").strip()
    if not path:
        raise KeyboardInterrupt
    selected = Path(path).expanduser().resolve()
    config = load_profile(selected)
    if config.room_id != room["room_id"] or origin(config.api_url) != api_url:
        raise SetupError("That profile belongs to another room or API. Provision a new installation on the backend.")
    return selected, config


def save_launcher(path, command):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SetupError("Refusing to replace a symlink or non-file launcher.")
    if path.exists() and not path.read_text().startswith("#!/bin/sh\n# Generated by interface/setup.py\n"):
        raise SetupError("An unrelated launcher already exists; move it before running setup.")
    # Atomically replace only this generated launcher. No tokens are embedded.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".sensor-launch-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write("#!/bin/sh\n# Generated by interface/setup.py\n")
            handle.write("unset " + " ".join(ENV_KEYS) + "\n")
            handle.write(f"cd {shlex.quote(str(ROOT))} || exit 1\n")
            handle.write("exec " + shlex.join(command) + ' "$@"\n')
            handle.flush()
            temporary.chmod(0o700)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def default_backend():
    for name in ("Room-Capacity-Website", "Room-Capacity-App"):
        path = ROOT.parent / name
        if (path / "scripts/add-device.js").is_file():
            return path
    return ROOT.parent / "Room-Capacity-Website"


def select_microphone(camera, *, use_fzf=True):
    from src.get_data import list_microphones, resolve_microphone
    if sys.platform != "darwin":
        try:
            microphone = resolve_microphone(int(camera) if camera.isdecimal() else camera, None)
            print(f"Matched webcam microphone: input {microphone}.")
            return microphone
        except ValueError:
            pass
    # AVFoundation camera indices and CoreAudio microphone indices are unrelated.
    try:
        microphones = list_microphones()
    except Exception:
        hint = " On macOS, allow Microphone access for your terminal or IDE in System Settings → Privacy & Security." if sys.platform == "darwin" else ""
        raise SetupError("Could not list microphone inputs. Check the audio device and PortAudio installation." + hint) from None
    return choose("Microphone", [(f"{mic['name']} · input {mic['device']}", mic["device"]) for mic in microphones]
                  + [("System default microphone", "default")], use_fzf=use_fzf)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", help=f"Website origin (default: profile origin or {DEFAULT_API})")
    parser.add_argument("--env", type=Path, help="Existing private device profile, e.g. .env.laptop")
    parser.add_argument("--backend", type=Path, default=default_backend(), help="Trusted backend checkout for optional new-device registration")
    parser.add_argument("--no-fzf", action="store_true", help="Use the numbered fuzzy-search fallback")
    parser.add_argument("--configure-only", action="store_true", help="Save the launcher without offering to start capture")
    args = parser.parse_args(argv)
    started = False
    try:
        profile_path = args.env.expanduser().resolve() if args.env else None
        available = [(profile_path, load_profile(profile_path))] if profile_path else profiles()
        api_url = origin(args.api_url or (available[0][1].api_url if available else DEFAULT_API))
        print(f"Website: {api_url}")
        rooms = fetch_rooms(api_url)
        cameras = discover_cameras()
        if cameras:
            camera = choose("Webcam", [(f"{item['name']} · {item['device']}", str(item["device"])) for item in cameras], use_fzf=not args.no_fzf)
        else:
            if sys.platform == "darwin":
                print("No macOS cameras reported. Check Camera access for your terminal or IDE in System Settings → Privacy & Security.")
                prompt = "Enter a camera index (e.g. 0), or Enter to cancel: "
            else:
                prompt = "No accessible Linux webcams detected. Enter a camera index or /dev/video path: "
            value = input(prompt).strip()
            if not value:
                raise KeyboardInterrupt
            if not (value.isdecimal() or (sys.platform != "darwin" and value.startswith("/dev/"))):
                raise SetupError("No webcam selected. Check permissions and reconnect the camera.")
            camera = value
        # Importing hardware helpers never captures media. YOLO is loaded only
        # after the user elects to start the finished launcher.
        microphone = select_microphone(camera, use_fzf=not args.no_fzf)
        room = choose("Room", [(room_label(room), room) for room in rooms], fuzzy=True, use_fzf=not args.no_fzf)
        print(f"Selected room: {room_label(room)}")
        profile, config = select_profile(api_url, room, available, args.backend.expanduser().resolve(), use_fzf=not args.no_fzf)
        command = [sys.executable, str(ROOT / "interface/main.py"), "--upload", "--env", str(profile),
                   "--camera", camera, "--audio-device", str(microphone)]
        launcher = ROOT / "run-sensor.local.sh"
        save_launcher(launcher, command)
        print(f"\nReady: {display(config.device_id)} → {display(config.room_id)}")
        print(f"Saved launcher: {launcher}\nRun it again with: ./run-sensor.local.sh")
        if sys.platform == "darwin":
            print("macOS: allow Camera and Microphone access for your terminal or IDE when prompted. Rerun setup if devices are reconnected or reordered.")
        if not args.configure_only and input("Start uploading sensor readings now? [Y/n]: ").strip().lower() in ("", "y", "yes"):
            started = True
            return subprocess.call([str(launcher)])
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nSensor stopped." if started else "\nSetup cancelled. No sensor was started.")
        return 130
    except SetupError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, subprocess.TimeoutExpired):
        # Configuration/device errors must never echo secrets or child output.
        print("Setup failed. Check the room/API, device profile and hardware permissions.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
