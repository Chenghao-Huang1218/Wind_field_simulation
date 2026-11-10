"""Manage only the local SITL child created by the project's launcher."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Atomically publish a complete session or shutdown request record."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(record, stream)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def request_managed_sitl_shutdown(state_directory: Path) -> bool:
    """Request shutdown from a live launcher without signalling stored PIDs.

    Args:
        state_directory: State directory shared with the local SITL launcher.

    Returns:
        Whether a request was written for a currently managed session.
    """
    state_directory = state_directory.resolve()
    try:
        with (state_directory / "sitl.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                record = json.loads(
                    (state_directory / "sitl-session.json").read_text()
                )
                session_id = record.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    return False
                _write_record(
                    state_directory / "sitl-stop.request",
                    {"session_id": session_id},
                )
                return True
            # No launcher owns the lock, so any session record is stale.
            fcntl.flock(lock, fcntl.LOCK_UN)
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return False


def _stop_child(process: subprocess.Popen[Any]) -> None:
    """Allow PX4's POSIX SIGINT handler to stop this owned child first."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        print("SITL did not exit after SIGINT; terminating its child process.", flush=True)
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def run_managed_sitl(
    runtime_directory: Path,
    state_directory: Path,
    extra_arguments: Sequence[str] = (),
) -> int:
    """Run SITL and consume shutdown requests for this launcher session only.

    Args:
        runtime_directory: Runtime containing ``bin/px4`` and ``etc/``.
        state_directory: Writable directory for PX4 state and lifecycle records.
        extra_arguments: Additional PX4 command-line arguments.

    Returns:
        PX4's exit code, or zero after a requested orderly stop.

    Raises:
        FileNotFoundError: If the runtime is incomplete.
        RuntimeError: If another launcher owns the same state directory.
    """
    runtime_directory = runtime_directory.resolve()
    state_directory = state_directory.resolve()
    binary = runtime_directory / "bin/px4"
    startup = runtime_directory / "etc/init.d-posix/rcS"
    if not binary.is_file() or not startup.is_file():
        raise FileNotFoundError(f"Incomplete PX4 runtime: {runtime_directory}")
    state_directory.mkdir(parents=True, exist_ok=True)
    with (state_directory / "sitl.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"SITL launcher already active: {state_directory}") from error
        session_path = state_directory / "sitl-session.json"
        request_path = state_directory / "sitl-stop.request"
        request_path.unlink(missing_ok=True)
        # This random identifier isolates lifecycle requests; it does not seed or
        # affect any numerical simulation or experiment random generator.
        session_id = uuid.uuid4().hex
        environment = os.environ.copy()
        environment.update(PX4_SIM_MODEL="none_iris", PX4_SIM_FLOAT_STATE="1")
        environment["PATH"] = f"{binary.parent}{os.pathsep}{environment.get('PATH', '')}"
        process: subprocess.Popen[Any] | None = None
        requested_stop = False
        try:
            process = subprocess.Popen(
                [str(binary), str(runtime_directory / "etc"),
                 "-s", "etc/init.d-posix/rcS", *extra_arguments],
                cwd=state_directory, env=environment,
            )
            _write_record(session_path, {"session_id": session_id, "pid": process.pid})
            print(f"Managed PX4 SITL started (PID {process.pid}).", flush=True)
            while process.poll() is None:
                try:
                    request = json.loads(request_path.read_text())
                except (OSError, ValueError):
                    request = {}
                if isinstance(request, dict) and request.get("session_id") == session_id:
                    requested_stop = True
                    print("EXIT received; stopping this PX4 SITL instance.", flush=True)
                    _stop_child(process)
                    break
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            requested_stop = True
        finally:
            if process is not None:
                _stop_child(process)
            session_path.unlink(missing_ok=True)
            request_path.unlink(missing_ok=True)
        return 0 if requested_stop else int(process.returncode)
