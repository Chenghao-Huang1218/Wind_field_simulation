"""Observation logs must support reproducible, explicit comparison windows."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from wind_field_simulation.run_logging import summarize_wind_run


def write_log(path: Path) -> None:
    """Emit independent, hand-calculable one-second fixture observations."""
    rows = [{"type": "metadata", "wind_backend": "analytic"}]
    for index in range(8):
        rows.append({
            "type": "sample", "sim_time_s": index,
            "wind_enabled": True, "trajectory": "none",
            "flight_state": "FLYING", "position_world_m": [3, 4, 2],
            "target_world_m": [0, 0, 1],
            "configured_wind_world_m_s": [1, 0, 0],
            "force_world_n": [0.1, 0, 0],
            "arm_joint_deviation": {"joint1": 0.1, "endleft": 0.001},
            "acceleration_clipped": False,
            "outside_inspection_domain": False,
            "wind_compute_wall_s": 0.0005, "iteration_wall_s": 0.001,
            "last_error": "none",
        })
    rows.append({"type": "closed", "exception": None,
                 "last_error": "none"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_summary_metrics_use_same_target_and_explicit_window(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_log(path)
    result = summarize_wind_run(path, settle_s=1, duration_s=3)
    assert result["position_xy_rmse_m"] == pytest.approx(5)
    assert result["position_3d_rmse_m"] == pytest.approx(26**0.5)
    assert result["height_rmse_m"] == pytest.approx(1)
    assert result["sample_count"] == 3
    assert result["window_s"] == [1, 4]
    assert result["joint_max_abs_deviation"]["endleft"] == 0.001
    assert result["force_z_max_abs_n"] == 0
    assert result["wind_compute_p95_ms"] == pytest.approx(0.5)
    assert result["closed_cleanly"]


def test_summary_refuses_short_window_and_reports_incomplete_log(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_log(path)
    with pytest.raises(ValueError, match="window"):
        summarize_wind_run(path, duration_s=60)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    result = summarize_wind_run(path, settle_s=0, duration_s=3)
    assert not result["closed_cleanly"]


def test_summary_rejects_mixed_wind_cases(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_log(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[3]["configured_wind_world_m_s"] = [2, 0, 0]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="wind"):
        summarize_wind_run(path, settle_s=0, duration_s=4)


def test_summary_cli_runs_on_saved_observations(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_log(path)
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_wind_run.py"
    result = subprocess.run([
        sys.executable, str(script), str(path), "--settle-s", "1",
        "--duration-s", "3",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["position_xy_rmse_m"] == pytest.approx(5)
