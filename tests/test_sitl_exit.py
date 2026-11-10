"""Offline lifecycle regressions; never launch a real PX4 executable."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from wind_field_simulation import simulation


@pytest.mark.parametrize("exit_requested", [True, False])
def test_main_requests_local_shutdown_only_after_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exit_requested: bool,
) -> None:
    """Closing via EXIT must notify the managed SITL after app cleanup."""
    events: list[str] = []

    class OfflineApp:
        """Substitute only the graphical app and transport for this CLI test."""

        def __init__(self, _config: Any) -> None:
            self.exit_requested = exit_requested

        def run(self) -> None:
            """Represent completed window and transport cleanup."""
            events.append("closed")

    def request_shutdown(state_directory: Path) -> bool:
        """Record the lifecycle request without contacting a running instance."""
        assert state_directory == tmp_path
        events.append("stop")
        return True

    monkeypatch.setenv("PX4_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(simulation, "WindSimulationApp", OfflineApp)
    monkeypatch.setattr(
        simulation, "request_managed_sitl_shutdown", request_shutdown, raising=False,
    )
    assert simulation.main(["--no-viewer"]) == 0
    assert events == (["closed", "stop"] if exit_requested else ["closed"])


def test_managed_launcher_stops_only_its_child(tmp_path: Path) -> None:
    """Exercise the actual supervisor with a harmless fake SITL process."""
    lifecycle = importlib.import_module("wind_field_simulation.sitl_process")
    runtime = tmp_path / "runtime"
    binary = runtime / "bin/px4"
    binary.parent.mkdir(parents=True)
    (runtime / "etc/init.d-posix").mkdir(parents=True)
    (runtime / "etc/init.d-posix/rcS").write_text("# offline fixture\n")
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
        "Path('ready').write_text('ready')\n"
        "while True: time.sleep(0.1)\n"
    )
    binary.chmod(0o755)
    state = tmp_path / "state"
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    launcher = Path(__file__).resolve().parents[1] / "scripts/run_px4_sitl.py"
    manager = subprocess.Popen([
        sys.executable, str(launcher), "--runtime-dir", str(runtime),
        "--state-dir", str(state),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 5.0
        while not (state / "ready").exists() and time.monotonic() < deadline:
            assert manager.poll() is None
            time.sleep(0.02)
        assert (state / "ready").exists(), "Offline child did not start"
        duplicate = subprocess.run(
            [sys.executable, str(launcher), "--runtime-dir", str(runtime),
             "--state-dir", str(state)],
            capture_output=True, text=True, timeout=5.0,
        )
        assert duplicate.returncode == 1
        assert "already active" in duplicate.stderr
        (state / "sitl-stop.request").write_text(json.dumps({"session_id": "old-session"}))
        time.sleep(0.25)
        assert manager.poll() is None, "An old session request stopped the new child"
        assert lifecycle.request_managed_sitl_shutdown(state)
        assert manager.wait(timeout=5.0) == 0
        assert unrelated.poll() is None
        assert not (state / "sitl-session.json").exists()
        assert not lifecycle.request_managed_sitl_shutdown(state)
    finally:
        if manager.poll() is None:
            lifecycle.request_managed_sitl_shutdown(state)
            manager.terminate()
            manager.wait(timeout=5.0)
        unrelated.terminate()
        unrelated.wait(timeout=5.0)
        if manager.stdout:
            manager.stdout.close()


def test_stale_session_does_not_trigger_process_signals(tmp_path: Path) -> None:
    """A stale record without a live launcher must not create a stop request."""
    lifecycle = importlib.import_module("wind_field_simulation.sitl_process")
    (tmp_path / "sitl-session.json").write_text(json.dumps({"session_id": "stale"}))
    (tmp_path / "sitl.lock").touch()
    assert not lifecycle.request_managed_sitl_shutdown(tmp_path)
    assert not (tmp_path / "sitl-stop.request").exists()
