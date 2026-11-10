#!/usr/bin/env python3
"""Summarize one explicit window of a saved manual SITL observation log."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.run_logging import summarize_wind_run  # noqa: E402


def main() -> int:
    """Print JSON metrics; reject incomplete/mixed comparison windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--settle-s", type=float, default=5)
    parser.add_argument("--duration-s", type=float, default=60)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--trajectory", default="none",
                        choices=("none", "circle", "ellipse", "figure8", "square"))
    args = parser.parse_args()
    try:
        summary = summarize_wind_run(
            args.log, args.settle_s, args.duration_s,
            args.baseline, args.trajectory,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
