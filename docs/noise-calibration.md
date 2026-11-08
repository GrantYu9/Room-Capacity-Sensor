# Qualitative noise calibration

The website displays **Quiet**, **Moderate** and **Lively**. This sensor measures
the full recording's RMS digital amplitude, converts it to dBFS, and applies a
versioned piecewise mapping. It uploads `noise_unit: "relative"`, not dBA.

| Category | Bundled synthetic profile | Website relative scale |
|---|---|---|
| Quiet | below -47.5 dBFS | below 0.25 |
| Moderate | -47.5 through below -27 dBFS | 0.25 through below 0.6 |
| Lively | -27 dBFS and above | 0.6 through 1 |

The mapping interpolates through `(-120, 0)`, `(-47.5, 0.25)`, `(-27, 0.6)`,
`(0, 1)` and clamps to [0,1]. Boundary values belong to the higher category,
matching both the frontend and server's community noise logic. The relative
number is a category-aligned index, not a percentage of physical loudness or
sound pressure. Local JSON preserves RMS, raw `decibels` (dBFS), the category and
the profile ID for diagnostics. Only the relative index/unit go into readings.

## Generated fixtures

```sh
.venv/bin/python testing/generate_noise.py
```

This reproducibly creates 15 two-second, 16 kHz float WAVs using seed 42:
six calibration examples (two per band), and nine checking examples (three per
band). Different phases/noise generate different waveforms. Fixtures contain
harmonic tones and noise at specified RMS levels; they are not recordings of
people. Output is `data/testing/noise-synthetic-v1/`, including two manifests,
WAV hashes, fitted profile, dependency versions and `results.json`.

The fit chooses the midpoint between the loudest quiet and quietest moderate
example, and likewise between moderate and lively. Each band needs at least
two examples; overlapping labelled amplitude ranges fail rather than inventing
a confident separation. The bundled thresholds match the generated fit.

All nine checking fixtures should reach the intended website category. Tests
also cover silence, exact boundaries and overlapping calibration labels. This
is a deterministic software check: labels were designed from digital amplitude,
so success does **not** estimate human perception or real-room accuracy. No
model is trained, and no claim about calibrated dBA is made.

## Calibrate an actual installation

Keep microphone, gain, placement and automatic gain control settings fixed.
Collect multiple representative recordings and have people label the room's
noise as quiet/moderate/lively. Keep separate recordings for checking after
fitting; do not use the same recording or overlapping chunks for both. Include
HVAC, speech, music and transient cases typical of that room. No live recording
is performed by the generated fixture script.

Create a manifest next to your WAVs:

```json
{
  "samples": [
    {"file": "quiet-1.wav", "band": "quiet"},
    {"file": "quiet-2.wav", "band": "quiet"},
    {"file": "moderate-1.wav", "band": "moderate"},
    {"file": "moderate-2.wav", "band": "moderate"},
    {"file": "lively-1.wav", "band": "lively"},
    {"file": "lively-2.wav", "band": "lively"}
  ]
}
```

```sh
.venv/bin/python -m interface.calibrate_noise \
  --manifest /private/calibration/labels.json \
  --output /private/calibration/library-01.json --id library-01-fixed-gain-v1 \
  --provenance "Labelled room recordings; microphone and gain recorded separately"
.venv/bin/python interface/main.py --upload --env .env.sensor \
  --audio-device YOUR-MIC-ID \
  --noise-calibration /private/calibration/library-01.json
```

Check the held-out WAVs using `audio_level(path)` and
`NoiseCalibration.load(profile_path).classify(dbfs)` from `src.noise`; compare
their bands with human labels and record errors. If bands overlap, simple RMS
cannot support those labels with this microphone setup. Investigate gain/AGC or
a richer acoustic feature rather than forcing a fit. Recalibrate after changing
hardware, gain or position. Physical dBA needs a measured reference and the
appropriate acoustic weighting; generated audio cannot supply that reference.
