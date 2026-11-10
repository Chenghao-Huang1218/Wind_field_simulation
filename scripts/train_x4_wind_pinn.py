#!/usr/bin/env python3
"""Train a separate CPU pilot candidate; never install it in the simulator."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training.train_v1 import train_pilot  # noqa: E402


def main() -> None:
    """Run the predeclared seed-42, max-200-epoch configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_pilot(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
