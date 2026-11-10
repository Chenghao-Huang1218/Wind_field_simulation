#!/usr/bin/env python3
"""Inspect a frozen drag reference offline; no dataset, training or SITL."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training.reference_aerodynamics import (  # noqa: E402
    paired_wind_labels,
    reference_wrench_body,
)
from wind_field_simulation.training.xml_parameters import (  # noqa: E402
    PROJECT_ROOT,
    load_reference_bundle,
)


def _quaternions(
    roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray,
) -> np.ndarray:
    """Construct B-to-W WXYZ for Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return np.column_stack((cr * cp * cy + sr * sp * sy,
                            sr * cp * cy - cr * sp * sy,
                            cr * sp * cy + sr * cp * sy,
                            cr * cp * sy - sr * sp * cy))


def inspect_reference(
    reference: dict[str, Any], samples: int, repeats: int,
) -> dict[str, Any]:
    """Check bounded states, rest-wind grid and CPU direct-evaluation latency.

    Args:
        reference: Verified self-contained reference artifact.
        samples: Number of seeded inspection states, separate from training data.
        repeats: Number of warmed scalar pair timing calls.

    Returns:
        JSON-compatible behavior, source and timing metrics; no training data.
    """
    rng = np.random.default_rng(42)
    angle = rng.uniform(-np.pi, np.pi, samples)
    speed = 3 * np.sqrt(rng.uniform(0, 1, samples))
    velocity = np.column_stack((speed * np.cos(angle), speed * np.sin(angle),
                                rng.uniform(-1, 1, samples)))
    wind_angle = rng.uniform(-np.pi, np.pi, samples)
    wind_speed = 3 * np.sqrt(rng.uniform(0, 1, samples))
    wind = np.column_stack((
        wind_speed * np.cos(wind_angle), wind_speed * np.sin(wind_angle),
        np.zeros(samples),
    ))
    roll, pitch = rng.uniform(-np.deg2rad(20), np.deg2rad(20), (2, samples))
    yaw = rng.uniform(-np.pi, np.pi, samples)
    quaternion = _quaternions(roll, pitch, yaw)
    direction = rng.normal(size=(samples, 3))
    omega = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    omega *= (1.5 * rng.uniform(0, 1, samples)**(1 / 3))[:, None]
    paired = paired_wind_labels(velocity, quaternion, omega, wind, reference)
    zero = paired_wind_labels(velocity, quaternion, omega, np.zeros(3), reference)
    # Test the complete wrench with air-relative translational velocity in B.
    air_body = rng.uniform(-6, 6, (samples, 3))
    force, moment = reference_wrench_body(air_body, omega, reference)
    power = np.sum(force * air_body + moment * omega, axis=-1)
    label = paired["wind_increment_acceleration_world_m_s2"]
    if not all(np.all(np.isfinite(value)) for value in paired.values()):
        raise ValueError("Nonfinite reference output")
    if (np.max(power) > 1e-10 or np.any(label[:, 2] != 0)
            or np.any(zero["wind_increment_acceleration_world_m_s2"] != 0)):
        raise ValueError("Reference physical property check failed")
    grid = []
    for magnitude in np.linspace(0, 3, 7):
        for degrees in range(0, 360, 15):
            theta = np.deg2rad(degrees)
            current_wind = magnitude * np.array([
                np.cos(theta), np.sin(theta), 0,
            ])
            result = paired_wind_labels(
                np.zeros(3), [1, 0, 0, 0], np.zeros(3),
                current_wind, reference,
            )
            increment = result["wind_increment_acceleration_world_m_s2"]
            grid.append({
                "wind_speed_m_s": float(magnitude),
                "wind_direction_deg": degrees, "state": "rest_identity",
                "force_world_n": result["force_wind_world_n"].tolist(),
                "increment_world_m_s2": increment.tolist(),
            })
    opposing = paired_wind_labels(
        [-3, 0, 0], [1, 0, 0, 0], [0, 0, 0], [3, 0, 0], reference,
    )
    grid.append({
        "wind_speed_m_s": 3, "wind_direction_deg": 0,
        "state": "root_velocity_minus3x",
        "force_world_n": opposing["force_wind_world_n"].tolist(),
        "increment_world_m_s2": opposing[
            "wind_increment_acceleration_world_m_s2"
        ].tolist(),
    })
    # Include validation and both full evaluations in timings; no model loading.
    for _ in range(20):
        paired_wind_labels(velocity[0], quaternion[0], omega[0], wind[0], reference)
    elapsed = np.empty(repeats)
    for index in range(repeats):
        start = perf_counter_ns()
        paired_wind_labels(velocity[index % samples], quaternion[index % samples],
                           omega[index % samples], wind[index % samples], reference)
        elapsed[index] = (perf_counter_ns() - start) / 1000
    batch = []
    for _ in range(10):
        start = perf_counter_ns()
        paired_wind_labels(velocity, quaternion, omega, wind, reference)
        batch.append((perf_counter_ns() - start) / 1000)
    label_norm = np.linalg.norm(label, axis=-1)
    full_z = paired["force_wind_world_n"][:, 2]
    zero_increment = zero["wind_increment_acceleration_world_m_s2"]
    return {
        "kind": "offline_reference_inspection_v1", "seed": 42,
        "state_samples": samples,
        "reference_sha256": reference["reference_sha256"],
        "environment": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
        },
        "inspection_state_ranges": {
            "horizontal_velocity_max_m_s": float(np.max(speed)),
            "vertical_velocity_minmax_m_s": [
                float(velocity[:, 2].min()), float(velocity[:, 2].max()),
            ],
            "wind_speed_max_m_s": float(np.max(wind_speed)),
            "angular_speed_max_rad_s": float(
                np.linalg.norm(omega, axis=-1).max()
            ),
        },
        "label_norm_minmax_m_s2": [
            float(label_norm.min()), float(label_norm.max()),
        ],
        "full_force_wind_z_minmax_n": [float(full_z.min()), float(full_z.max())],
        "label_z_max_abs_m_s2": float(np.max(np.abs(label[:, 2]))),
        "zero_wind_max_abs_increment_m_s2": float(np.abs(zero_increment).max()),
        "max_full_drag_power_w": float(power.max()),
        "scalar_pair_timing_us": {
            "median": float(np.median(elapsed)),
            "p95": float(np.percentile(elapsed, 95)),
            "p99": float(np.percentile(elapsed, 99)),
            "max": float(elapsed.max()), "repeats": repeats,
        },
        "batch_pair_timing_us": {
            "median_total": float(np.median(batch)),
            "median_per_state": float(np.median(batch) / samples),
            "repeats": len(batch),
        },
        "rest_wind_grid": grid,
        "decision": (
            "Retain analytical reference as candidate baseline; "
            "no measured need for a PINN. Timing is an offline microbenchmark, "
            "not SITL real-time validation."
        ),
    }


def main() -> int:
    """Save inspection metrics beside the frozen reference and print key checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path,
        default=(PROJECT_ROOT / "outputs/pinn/x4_950_wind_v1"
                 / "reference_config.json"),
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--timing-repeats", type=int, default=1000)
    args = parser.parse_args()
    if args.samples < 1 or args.timing_repeats < 1:
        parser.error("samples and timing-repeats must be positive")
    reference = load_reference_bundle(args.reference)
    report = inspect_reference(reference, args.samples, args.timing_repeats)
    module_dir = PROJECT_ROOT / "src/wind_field_simulation/training"
    report["implementation_sha256"] = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (
            Path(__file__).resolve(), module_dir / "reference_aerodynamics.py",
            module_dir / "xml_parameters.py",
        )
    }
    output = args.reference.parent / "reference_check.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Offline states={args.samples}; "
          f"wind grid={len(report['rest_wind_grid'])}; seed=42")
    print("Zero-wind increment="
          f"{report['zero_wind_max_abs_increment_m_s2']}; "
          f"label Z={report['label_z_max_abs_m_s2']}")
    print(f"Full drag max power={report['max_full_drag_power_w']:.6g} W")
    print(f"Scalar pair timing (us): {report['scalar_pair_timing_us']}")
    print(report["decision"])
    print(f"Report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
