"""Exercise the retained PX4 C++ receiver without launching SITL."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_local_position_pairing_contract(tmp_path: Path) -> None:
    """Reject stale/mismatched samples and preserve exact float coordinates."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the PX4 receiver contract test")
    project = Path(__file__).resolve().parents[1]
    receiver = (
        project / "third_party/PX4-Autopilot/src/modules/simulation"
        / "simulator_mavlink"
    )
    if not (receiver / "LocalPositionFeedback.hpp").is_file():
        pytest.skip(
            "PX4 source omitted from runtime-only distribution; "
            "receiver contract requires the full source checkout"
        )
    executable = tmp_path / "feedback-contract"
    subprocess.run(
        [
            compiler, "-std=c++14", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-I", str(receiver),
            str(project / "tests/cpp/test_local_position_feedback.cpp"),
            "-o", str(executable),
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=5,
    )
