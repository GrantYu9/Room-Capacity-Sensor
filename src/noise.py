"""Versioned digital-level mapping to the website's qualitative noise bands."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


BANDS = ("quiet", "moderate", "lively")
DEFAULT_PROFILE = Path(__file__).with_name("noise.synthetic.json")


def audio_level(audio, sample_rate=None):
    if isinstance(audio, (str, Path)):
        audio, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
    if sample_rate is None or not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("A positive audio sample_rate is required")
    audio = np.asarray(audio)
    if audio.ndim not in (1, 2) or audio.size == 0:
        raise ValueError("Audio must contain samples, optionally with channels")
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError("Audio samples must be normalized floating point values")
    audio = audio.astype(np.float64)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError("Audio must contain finite normalized samples in [-1, 1]")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return {"rms": rms, "decibels": float(20 * np.log10(max(rms, 1e-6))),
            "audio_duration_seconds": len(audio) / sample_rate, "sample_rate": sample_rate}


@dataclass(frozen=True)
class NoiseCalibration:
    profile_id: str
    quiet_max_dbfs: float
    moderate_max_dbfs: float
    provenance: str

    def __post_init__(self):
        if not self.profile_id or not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("Noise calibration needs an id and provenance")
        if not (-120 < self.quiet_max_dbfs < self.moderate_max_dbfs < 0):
            raise ValueError("Noise thresholds must be ordered between -120 and 0 dBFS")

    @classmethod
    def load(cls, path=DEFAULT_PROFILE):
        data = json.loads(Path(path).read_text())
        return cls(**{key: data[key] for key in ("profile_id", "quiet_max_dbfs", "moderate_max_dbfs", "provenance")})

    def classify(self, dbfs):
        if not math.isfinite(dbfs):
            raise ValueError("Noise level must be finite")
        # Align exactly with the website's <0.25 / <0.6 band boundaries.
        anchors = [(-120.0, 0.0), (self.quiet_max_dbfs, 0.25),
                   (self.moderate_max_dbfs, 0.6), (0.0, 1.0)]
        level = float(np.interp(dbfs, *zip(*anchors)))
        band = "quiet" if dbfs < self.quiet_max_dbfs else "moderate" if dbfs < self.moderate_max_dbfs else "lively"
        return {"noise_level": level, "noise_unit": "relative", "noise_band": band,
                "noise_calibration": self.profile_id}


def fit_profile(manifest_path, profile_id, provenance):
    """Fit separated, labelled RMS examples; held-out examples must be separate."""
    path = Path(manifest_path)
    samples = json.loads(path.read_text())["samples"]
    groups = {band: [] for band in BANDS}
    for sample in samples:
        if sample["band"] not in groups:
            raise ValueError("Unknown noise band in calibration manifest")
        groups[sample["band"]].append(audio_level(path.parent / sample["file"])["decibels"])
    if any(len(values) < 2 for values in groups.values()):
        raise ValueError("Calibration requires at least two examples per band")
    boundaries = []
    for low, high in zip(BANDS, BANDS[1:]):
        lower, upper = max(groups[low]), min(groups[high])
        if lower >= upper:
            raise ValueError("Labelled bands overlap in RMS; gain or labels need checking")
        boundaries.append((lower + upper) / 2)
    return NoiseCalibration(profile_id, *boundaries, provenance)
