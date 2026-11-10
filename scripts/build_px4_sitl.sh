#!/usr/bin/env bash
# Compile only; this script does not launch PX4 or a simulation.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${PROJECT_ROOT}/scripts/prepare_px4_snapshot.py"
cd -- "${PROJECT_ROOT}/third_party/PX4-Autopilot"
# All submodule working files are already included. Do not fetch replacements.
export GIT_SUBMODULES_ARE_EVIL=1
# Build the host executable, not the default "all" target: the latter also
# downloads/builds Gazebo plugins that the MuJoCo MAVLink bridge does not use.
make px4_sitl_default px4 "$@"
