"""Optional observation-only JSONL output for manual closed-loop comparisons."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class WindRunLogger:
    """Record metadata, synchronized 50 Hz samples and a termination marker."""

    PERIOD_S = 0.02

    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        """Create an exclusive log so prior experiments cannot be overwritten.

        Args:
            path: New JSONL path; parent directories are created if needed.
            metadata: Backend, model provenance and shared run settings.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        self.last_time_s = -float("inf")
        self.sample_count = 0
        try:
            self._write({**metadata, "type": "metadata",
                         "log_period_s": self.PERIOD_S})
            self.stream.flush()
        except BaseException:
            self.stream.close()
            raise

    def _write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")

    def due(self, sim_time_s: float) -> bool:
        """Return whether the pre-step simulation timestamp needs a sample."""
        return sim_time_s - self.last_time_s >= self.PERIOD_S - 1e-9

    def record(self, sample: dict[str, Any]) -> None:
        """Persist an already synchronized pre-step observation.

        Args:
            sample: State and force from the same simulation instant, with
                timing of the iteration that applied that force.
        """
        self._write({**sample, "type": "sample"})
        self.last_time_s = float(sample["sim_time_s"])
        self.sample_count += 1

    def close(self, termination: dict[str, Any]) -> None:
        """Flush a termination record and close the file, idempotently."""
        if self.stream.closed:
            return
        try:
            self._write({**termination, "type": "closed",
                         "sample_count": self.sample_count})
        finally:
            self.stream.close()

def summarize_wind_run(
    path: Path, settle_s: float = 5.0, duration_s: float = 60.0,
    baseline: bool = False, trajectory: str = "none",
) -> dict[str, Any]:
    """Summarize an explicit continuous window of observed flight samples.

    Args:
        path: JSONL log emitted by the application, never synthetic labels.
        settle_s: Offset after the first matching FLYING/trajectory/wind sample.
        duration_s: Required complete window length, in simulation seconds.
        baseline: Select wind OFF samples; otherwise select wind ON samples.
        trajectory: Exact trajectory name, e.g. none, circle or figure8.

    Returns:
        Position errors relative to each simultaneous target, force/domain/
        joint/latency diagnostics, and an explicit clean-close flag. Metrics
        describe this run; no cross-backend accuracy claim is inferred.

    Raises:
        ValueError: The window is short, interrupted or contains mixed winds.
    """
    if (not np.isfinite(settle_s) or settle_s < 0
            or not np.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("Window offset must be >=0 and duration positive")
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("type") != "metadata":
        raise ValueError("Missing log metadata")
    samples = [row for row in rows if row["type"] == "sample"]
    eligible = [row for row in samples
                if row["flight_state"] == "FLYING"
                and row["trajectory"] == trajectory
                and row["wind_enabled"] == (not baseline)]
    if not eligible:
        raise ValueError("No matching flight samples for comparison window")
    start = eligible[0]["sim_time_s"] + settle_s
    end = start + duration_s
    selected = [row for row in eligible
                if start <= row["sim_time_s"] < end]
    all_window = [row for row in samples if start <= row["sim_time_s"] < end]
    if (not selected or eligible[-1]["sim_time_s"] < end
            or len(selected) != len(all_window)):
        raise ValueError("Comparison window incomplete or interrupted")
    winds = {tuple(row["configured_wind_world_m_s"] or [])
             for row in selected}
    if len(winds) != 1:
        raise ValueError("Mixed wind configurations in comparison window")
    errors = np.array([np.subtract(row["position_world_m"],
                                   row["target_world_m"]) for row in selected])
    forces = np.array([row["force_world_n"] for row in selected])
    norm = np.linalg.norm(errors, axis=-1)
    joint_names = set().union(*(row["arm_joint_deviation"] for row in selected))
    footer = rows[-1] if rows[-1]["type"] == "closed" else None
    clean = bool(
        footer is not None and footer.get("exception") is None
        and footer.get("last_error", "none") == "none"
        and footer.get("flight_state") != "EMERGENCY_STOP"
        and all(row["last_error"] == "none" for row in samples)
    )
    return {
        "metadata": rows[0], "window_s": [start, end],
        "baseline_wind_off": baseline, "trajectory": trajectory,
        "configured_wind_world_m_s": selected[0]["configured_wind_world_m_s"],
        "sample_count": len(selected), "closed_cleanly": clean,
        "termination": footer,
        "position_3d_rmse_m": float(np.sqrt(np.mean(norm**2))),
        "position_xy_rmse_m": float(np.sqrt(np.mean(
            np.sum(errors[:, :2]**2, axis=-1),
        ))),
        "position_3d_peak_m": float(norm.max()),
        "height_rmse_m": float(np.sqrt(np.mean(errors[:, 2]**2))),
        "height_peak_abs_m": float(np.abs(errors[:, 2]).max()),
        "force_z_max_abs_n": float(np.abs(forces[:, 2]).max()),
        "force_norm_peak_n": float(np.linalg.norm(forces, axis=-1).max()),
        "clipped_fraction": float(np.mean([
            row["acceleration_clipped"] for row in selected
        ])),
        "outside_inspection_domain_fraction": float(np.mean([
            row["outside_inspection_domain"] for row in selected
        ])),
        "joint_max_abs_deviation": {
            name: max(abs(row["arm_joint_deviation"][name]) for row in selected)
            for name in sorted(joint_names)
        },
        "wind_compute_p95_ms": float(np.percentile([
            row["wind_compute_wall_s"] * 1000 for row in selected
        ], 95)),
        "iteration_compute_p95_ms": float(np.percentile([
            row["iteration_wall_s"] * 1000 for row in selected
        ], 95)),
    }
