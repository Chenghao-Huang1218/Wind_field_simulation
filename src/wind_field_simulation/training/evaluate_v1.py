"""Pilot-only raw error, zero-wind and CPU latency diagnostics."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray

from .dataset_v1 import file_sha256, load_dataset
from .model_v1 import X4WindAeroV1, load_checkpoint, predict_increment
from .reference_aerodynamics import paired_wind_labels
from .sampling import body_to_world_rotation
from .xml_parameters import load_reference_bundle


def error_metrics(prediction: NDArray, label: NDArray, nonzero_wind: NDArray) -> dict[str, Any]:
    """Report raw XY acceleration errors and nonzero-wind normalized RMSE.

    Args:
        prediction: World XY-projected accelerations [N,3], m/s².
        label: Reference acceleration increments [N,3], m/s².
        nonzero_wind: Boolean [N]; excludes zero wind from relative denominator.

    Returns:
        Absolute axis/vector errors, reference size, and a relative aggregate
        only when its denominator is numerically informative. No EMA or clip.
    """
    error = np.asarray(prediction)[:, :2] - np.asarray(label)[:, :2]
    magnitude = np.linalg.norm(error, axis=1)
    denominator = (float(np.sqrt(np.mean(np.sum(label[nonzero_wind, :2]**2, axis=1))))
                   if np.any(nonzero_wind) else 0.)
    numerator = (float(np.sqrt(np.mean(np.sum(error[nonzero_wind]**2, axis=1))))
                 if np.any(nonzero_wind) else 0.)
    return {
        "count": len(error), "axis_rmse_m_s2": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "vector_error_mean_m_s2": float(np.mean(magnitude)),
        "vector_error_p95_m_s2": float(np.percentile(magnitude, 95)),
        "vector_error_max_m_s2": float(np.max(magnitude)),
        "reference_nonzero_vector_rms_m_s2": denominator,
        "normalized_nonzero_vector_rmse": numerator/denominator if denominator > 1e-8 else None,
    }


def _timings_ms(values: list[int]) -> dict[str, float]:
    times = np.array(values) / 1e6
    return {"mean_ms": float(np.mean(times)), "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "max_ms": float(np.max(times))}


def benchmark_pairs(
    network: X4WindAeroV1, data: dict[str, NDArray], reference: dict[str, Any],
    repeats: int = 1000,
) -> dict[str, Any]:
    """Time warmed nonzero scalar pairs, alternating order on the same CPU.

    Args:
        network: CPU float32 model in eval mode, no compilation.
        data: Pilot state arrays; only nonzero-wind heldout rows are timed.
        reference: Float64 nominal analytical teacher.
        repeats: Measured calls per backend after 50 warmup calls each.

    Returns:
        Public-call timings, including validation, B/W transforms and XY
        projection. Teacher also computes diagnostic torque; no filter/clip.
        This compares actual implementations at their specified precisions,
        not matched-dtype numerical kernels or full SITL loop latency.
    """
    ids = np.flatnonzero((data["split"] == 2) & np.any(data["wind_world"] != 0, axis=1))

    def analytical(index: int) -> NDArray:
        return paired_wind_labels(
            data["velocity_world"][index:index+1], data["quaternion_wxyz"][index:index+1],
            data["omega_body"][index:index+1], data["wind_world"][index:index+1], reference,
        )["wind_increment_acceleration_world_m_s2"]

    def surrogate(index: int) -> NDArray:
        return predict_increment(
            network, data["velocity_world"][index:index+1],
            body_to_world_rotation(data["quaternion_wxyz"][index:index+1]),
            data["omega_body"][index:index+1], data["wind_world"][index:index+1],
        )

    calls: dict[str, Callable[[int], NDArray]] = {"analytic_float64": analytical,
                                                "pinn_float32": surrogate}
    durations: dict[str, list[int]] = {name: [] for name in calls}
    for repeat in range(50 + repeats):
        index = int(ids[repeat % len(ids)])
        order = list(calls) if repeat % 2 else list(reversed(calls))
        for name in order:
            start = perf_counter_ns()
            calls[name](index)
            elapsed = perf_counter_ns() - start
            if repeat >= 50:
                durations[name].append(elapsed)
    return {"device": "CPU", "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(), "warmup_calls_each": 50,
            "measured_calls_each": repeats, "batch_pairs": 1,
            "scope": "public pair call with frame conversion/projection, no EMA/clip; teacher includes torque",
            **{name: _timings_ms(values) for name, values in durations.items()}}


def evaluate_pilot(manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    """Evaluate a fixed candidate once, saving diagnostic holdout and timings.

    Args:
        manifest_path: Frozen nominal pilot dataset manifest.
        checkpoint_path: Best-validation candidate.pt.
        output_dir: New directory for evaluation.json and predictions.npz.

    Returns:
        Report path. This is not a formal final test or a flight validation.
    """
    data, manifest = load_dataset(manifest_path)
    network, saved = load_checkpoint(checkpoint_path)
    if (saved["dataset_manifest_sha256"] != manifest["manifest_sha256"]
            or saved["dataset_archive_sha256"] != manifest["archive_sha256"]
            or saved["reference_sha256"] != manifest["reference_sha256"]):
        raise ValueError("Checkpoint and dataset provenance differ")
    reference = load_reference_bundle(manifest_path.parent / manifest["reference_bundle"])
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction = predict_increment(network, data["velocity_world"],
                                    data["rotation_body_to_world"], data["omega_body"],
                                    data["wind_world"])
    label = data["delta_acceleration_world"]
    nonzero = np.any(data["wind_world"] != 0, axis=1)
    zero = predict_increment(network, data["velocity_world"],
                             data["rotation_body_to_world"], data["omega_body"],
                             np.zeros_like(data["wind_world"]))
    air = torch.tensor(data["air_velocity_body_pair"].reshape(-1, 3), dtype=torch.float32)
    omega = torch.tensor(np.repeat(data["omega_body"][:, None, :], 2, axis=1).reshape(-1, 3),
                         dtype=torch.float32)
    with torch.inference_mode():
        full_prediction = network(air, omega).numpy().reshape(-1, 2, 3)
        origin = network(torch.zeros(1, 3), torch.zeros(1, 3)).numpy()[0]
        # No bypass in this identity test: identical inputs in a single pair batch.
        duplicated = torch.tensor(np.repeat(data["air_velocity_body_pair"][:, 1:2, :],
                                              2, axis=1).reshape(-1, 3), dtype=torch.float32)
        identical = network(duplicated, omega).numpy().reshape(-1, 2, 3)
    split_reports = {}
    for split, name in enumerate(manifest["split_names"]):
        selected = data["split"] == split
        full_error = full_prediction[selected] - data["full_acceleration_body"][selected]
        split_reports[name] = {
            "pinn": error_metrics(prediction[selected], label[selected], nonzero[selected]),
            "zero_predictor": error_metrics(np.zeros_like(prediction[selected]),
                                             label[selected], nonzero[selected]),
            "full_body_axis_rmse_m_s2": np.sqrt(np.mean(full_error**2, axis=(0, 1))).tolist(),
        }
    heldout = data["split"] == 2
    wind_speed = np.linalg.norm(data["wind_world"][:, :2], axis=1)
    angle = np.mod(np.rad2deg(np.arctan2(data["wind_world"][:, 1],
                                         data["wind_world"][:, 0])), 360)
    tilt = np.max(np.abs(np.rad2deg(data["euler_rpy_rad"][:, :2])), axis=1)
    groups = {}
    for variable, values, edges in (
        ("wind_speed_m_s", wind_speed, [0, 1e-10, 1, 2, 3.00000001]),
        ("nonzero_wind_direction_deg", angle, list(range(0, 361, 30))),
        ("max_abs_roll_pitch_deg", tilt, [0, 10, 20.00000001]),
    ):
        group_metrics = {}
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            mask = heldout & (values >= lower) & (values < upper)
            if variable.startswith("nonzero"):
                mask &= nonzero
            if np.any(mask):
                group_metrics[f"[{lower},{upper})"] = error_metrics(
                    prediction[mask], label[mask], nonzero[mask])
        groups[variable] = group_metrics
    diagnostic = split_reports["pilot_holdout"]["pinn"]
    zero_rate = np.all(data["omega_body"] == 0, axis=1)
    work = np.sum(full_prediction[zero_rate] * data["air_velocity_body_pair"][zero_rate], axis=-1)
    report = {
        "kind": "x4_950_pinn_pilot_evaluation_v1",
        "status": "synthetic pilot diagnostic, not formal test or SITL",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "reference_sha256": manifest["reference_sha256"],
        "partitions": split_reports, "heldout_groups": groups,
        "provisional_gates": {
            "axis_rmse_limit_m_s2": .05, "vector_p95_limit_m_s2": .10,
            "normalized_nonzero_limit": .05,
            "absolute_pass": (max(diagnostic["axis_rmse_m_s2"]) <= .05
                              and diagnostic["vector_error_p95_m_s2"] <= .10),
            "normalized_pass": (diagnostic["normalized_nonzero_vector_rmse"] is not None
                                and diagnostic["normalized_nonzero_vector_rmse"] <= .05),
        },
        "zero_wind": {"states_checked": len(zero), "max_abs_increment_m_s2": float(np.max(np.abs(zero))),
                      "raw_identical_pair_difference_max_m_s2": float(np.max(np.abs(identical[:, 0] - identical[:, 1]))),
                      "all_projected_z_exact_zero": bool(np.all(prediction[:, 2] == 0)),
                      "full_acceleration_at_rest_body_m_s2": origin.tolist()},
        "zero_omega_full_work": {"pair_arms_checked": int(work.size),
                                 "positive_fraction_above_1e_minus_8": float(np.mean(work > 1e-8)),
                                 "max_m2_s3": float(np.max(work))},
        "latency": benchmark_pairs(network, data, reference),
        "sampling": {"seed": manifest["config"]["sampling_seed"],
                     "generation_seconds": manifest["generation_seconds"],
                     "array_memory_bytes": manifest["array_memory_bytes"],
                     "wind_speed_min_max_m_s": [float(wind_speed.min()), float(wind_speed.max())],
                     "zero_wind_pairs": int(np.sum(~nonzero)),
                     "zero_omega_pairs": int(np.sum(zero_rate)),
                     "speed_histogram_edges": [0, .5, 1, 1.5, 2, 2.5, 3.00000001],
                     "speed_histogram": np.histogram(wind_speed, bins=[0, .5, 1, 1.5, 2, 2.5, 3.00000001])[0].tolist(),
                     "direction_histogram_nonzero_30deg": np.histogram(angle[nonzero], bins=np.arange(0, 361, 30))[0].tolist(),
                     "force_abs_max_n": manifest["force_abs_max_n"],
                     "label_abs_max_m_s2": manifest["label_abs_max_m_s2"]},
    }
    np.savez_compressed(output_dir / "predictions.npz", prediction=prediction, label=label,
                         full_prediction=full_prediction, split=data["split"])
    path = output_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path
