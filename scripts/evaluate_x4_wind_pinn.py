#!/usr/bin/env python3
"""Evaluate a frozen pilot checkpoint without invoking PX4 or the simulator."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training.evaluate_v1 import evaluate_pilot  # noqa: E402


def main() -> None:
    """Save raw diagnostic errors, zero-wind identities and CPU pair timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate_pilot(args.manifest, args.checkpoint, args.output_dir))


if __name__ == "__main__":
    main()
