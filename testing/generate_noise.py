"""Reproduce digital-level calibration fixtures; no acoustic accuracy claim.

The target labels specify synthetic amplitude ranges. Distinct random waveforms
are used for fitting and checking, but both share a designed digital-level rule.
This verifies arithmetic/category plumbing, not perceptual generalization.
"""
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.noise import audio_level, fit_profile


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    sample_rate = 16000
    t = np.arange(sample_rate * 2) / sample_rate
    specifications = {
        "calibration": {"quiet": [-65, -55], "moderate": [-40, -34], "lively": [-20, -12]},
        "check": {"quiet": [-90, -60, -50], "moderate": [-44, -36, -30], "lively": [-24, -18, -10]},
    }
    for split, groups in specifications.items():
        samples = []
        for band, levels in groups.items():
            for index, dbfs in enumerate(levels):
                wave = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                           for f in [173, 431, 997]) + 0.15 * rng.normal(size=t.size)
                wave *= 10 ** (dbfs / 20) / np.sqrt(np.mean(wave ** 2))
                assert np.max(np.abs(wave)) < 1
                name = f"{split}-{band}-{index}.wav"
                sf.write(output / name, wave, sample_rate, subtype="FLOAT")
                samples.append({"file": name, "band": band, "target_dbfs": dbfs,
                                "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()})
        (output / f"{split}.json").write_text(json.dumps({"seed": 42, "samples": samples}, indent=2) + "\n")
    profile = fit_profile(output / "calibration.json", "synthetic-rms-v1",
                          "Generated digital-level fixtures; provisional, not acoustic/dBA calibration.")
    (output / "fitted-profile.json").write_text(json.dumps(asdict(profile), indent=2) + "\n")
    results = []
    for sample in json.loads((output / "check.json").read_text())["samples"]:
        level = audio_level(output / sample["file"])
        results.append({**sample, **level, **profile.classify(level["decibels"]), "expected_band": sample["band"]})
    passed = sum(row["expected_band"] == row["noise_band"] for row in results)
    report = {"purpose": "Software fixture checks only; these labels are designed from digital amplitude, not human room observations.",
              "seed": 42, "sample_rate": sample_rate, "calibration_examples": 6,
              "check_examples": len(results), "checks_passed": passed, "profile": asdict(profile),
              "dependencies": {key: version(key) for key in ["numpy", "soundfile"]}, "results": results}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"generated_wavs": 15, "checks_passed": passed, "checks": len(results), "output": str(output)}))
    assert passed == len(results)
    return profile, results


if __name__ == "__main__":
    generate(ROOT / "data/testing/noise-synthetic-v1")
