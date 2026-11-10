#!/usr/bin/env python3
"""Launch the package directly from an unpacked source tree."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.simulation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
