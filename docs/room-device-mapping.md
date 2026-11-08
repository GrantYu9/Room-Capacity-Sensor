# Assign this sensor to a website room

`ROOM_ID` is the exact existing website space ID. `DEVICE_ID` identifies this
installation of a camera/microphone pair. These are independent of OpenCV camera
indexes and PortAudio microphone IDs. One active primary sensor per room is the
current supported presentation: multiple readings are not summed.

From the trusted adjacent `Room-Capacity-App` checkout, run:

```sh
npm run db:confidence
npm run device:add -- --room EXACT_ROOM_ID --device pi-library-01 \
  --api-url https://YOUR-FINAL-API-ORIGIN --output .env.pi-library-01
```

Copy the resulting private file to this sensor as `.env.sensor` (mode 0600).
It contains `API_URL`, `ROOM_ID`, `DEVICE_ID`, `DEVICE_TOKEN`. Environment variables
override file values. Keep tokens and database URLs out of logs/Git. Only a
device token belongs on the sensor; database credentials stay on the backend.

Alternatively, run `.venv/bin/python interface/setup.py` for webcam selection,
microphone matching and fuzzy room search against the website API. It reuses a
profile only when both API origin and room ID match. For another room, it can
register a new installation through the trusted adjacent app checkout after you
choose that option, or use a separately provisioned profile. It never reassigns
or disables an existing device. The saved `run-sensor.local.sh` clears shell
credential overrides so it always uses the selected file.

```sh
.venv/bin/python interface/main.py --list-devices
.venv/bin/python interface/main.py --upload --env .env.sensor \
  --camera YOUR-CAMERA --audio-device YOUR-MIC-ID
```

The API derives device identity from the hashed bearer token and checks its room
assignment. The uploader verifies both IDs in the saved acknowledgement. A wrong
room is 403; inactive credentials are 401. Uploads never alter room metadata.
The complete backend guide is `docs/room-device-mapping.md` in the app repository,
including registration, retirement, data semantics and verification commands.

For a room move or replacement, stop and deactivate the old installation through
the backend, then issue a new device ID/token and use a new outbox. Keep the old
history and queued records for audit. Never relabel old measurements. The queue
is scoped by device/room and sends its original timestamps unchanged.

The bundled noise profile is provisional; see [noise calibration](noise-calibration.md).
`average_confidence` is the mean detected-person box score, or null without
detections. It is available in saved readings for a future occupancy trust policy,
not a calibrated probability of a correct count or a measure of noise accuracy.
