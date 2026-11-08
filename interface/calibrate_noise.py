"""Fit qualitative RMS boundaries from a manifest of labelled room WAVs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from src.noise import fit_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()
    profile = fit_profile(args.manifest, args.id, args.provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(profile), indent=2) + "\n")
    print(f"Saved {profile.profile_id}; use --noise-calibration {args.output}")


if __name__ == "__main__":
    main()
