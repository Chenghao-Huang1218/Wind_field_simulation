#!/usr/bin/env python3
"""Launch a project-owned PX4 SITL process with coordinated EXIT handling."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_field_simulation.sitl_process import run_managed_sitl  # noqa: E402


def main() -> int:
    """Resolve local runtime paths and supervise its single SITL child."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir", type=Path,
        default=Path(os.environ.get("PX4_RUNTIME_DIR", PROJECT_ROOT / "runtime/px4")),
    )
    parser.add_argument(
        "--state-dir", type=Path,
        default=Path(os.environ.get("PX4_STATE_DIR", PROJECT_ROOT / "outputs/px4")),
    )
    arguments, px4_arguments = parser.parse_known_args()
    try:
        return run_managed_sitl(arguments.runtime_dir, arguments.state_dir, px4_arguments)
    except (OSError, RuntimeError) as error:
        print(f"PX4 SITL launcher error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
