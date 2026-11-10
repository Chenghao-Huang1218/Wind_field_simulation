#!/usr/bin/env bash
# Supervise the copied or locally rebuilt SITL for coordinated panel EXIT.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "${PROJECT_ROOT}/scripts/run_px4_sitl.py" "$@"
