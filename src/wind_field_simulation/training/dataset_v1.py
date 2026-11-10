"""Immutable nominal reference pairs for a separate, small X4 PINN pilot."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .reference_aerodynamics import paired_wind_labels
from .sampling import sample_pilot_states
from .xml_parameters import PROJECT_ROOT, content_sha256, load_reference_bundle


def check_split_leakage(data: dict[str, NDArray]) -> dict[str, int]:
    """Reject shared groups or transformed core inputs across partitions.

    Args:
        data: Partition IDs, group IDs, body velocity pairs [N,2,3], omega [N,3].

    Returns:
        Duplicate audit counts. Identical wind/no-wind rows within a zero-wind
        pair are allowed; sharing any full core across partitions is rejected.
    """
    for group in np.unique(data["group_id"]):
        if len(np.unique(data["split"][data["group_id"] == group])) != 1:
            raise ValueError("Sampling group leaks across splits")
    core = np.concatenate((data["air_velocity_body_pair"],
                           np.repeat(data["omega_body"][:, None, :], 2, axis=1)), axis=2)
    owners: dict[tuple[float, ...], int] = {}
    duplicates = 0
    for row, split in zip(core.reshape(-1, 6), np.repeat(data["split"], 2), strict=True):
        # 1e-10 rounding also catches near-identical numerical representations.
        key = tuple(np.round(row, decimals=10))
        if key in owners:
            if owners[key] != split:
                raise ValueError("Transformed core input leaks across splits")
            duplicates += 1
        owners[key] = int(split)
    return {"cross_split_core_duplicates": 0,
            "within_split_repeated_core_evaluations": duplicates,
            "groups": len(np.unique(data["group_id"]))}


def build_pilot_dataset(
    reference: dict[str, Any], seed: int = 7,
    split_counts: tuple[int, int, int] = (2000, 500, 500),
) -> dict[str, NDArray]:
    """Evaluate complete forces and raw same-state increments without clipping.

    Args:
        reference: Verified nominal frozen geometry/assumptions.
        seed: Sampling seed; defaults to the pilot's 7.
        split_counts: Diagnostic partition sizes, not formal test set sizes.

    Returns:
        State, full force/moment pairs, full B acceleration and XY W increment.
    """
    data = sample_pilot_states(seed, split_counts)
    labels = paired_wind_labels(
        data["velocity_world"], data["quaternion_wxyz"], data["omega_body"],
        data["wind_world"], reference,
    )
    relative_world = np.stack((data["velocity_world"] - data["wind_world"],
                               data["velocity_world"]), axis=1)
    data["air_velocity_body_pair"] = np.einsum(
        "nji,nkj->nki", data["rotation_body_to_world"], relative_world
    )
    data.update(labels)
    data["full_acceleration_body"] = np.einsum(
        "nji,nkj->nki", data["rotation_body_to_world"],
        np.stack((labels["force_wind_world_n"], labels["force_no_wind_world_n"]), axis=1),
    ) / reference["mass_total_kg"]
    data["delta_acceleration_world"] = data.pop("wind_increment_acceleration_world_m_s2")
    check_split_leakage(data)
    if not all(np.isfinite(array).all() for array in data.values()):
        raise ValueError("Nonfinite pilot data")
    return data


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of a local artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_dataset(config_path: Path, output_dir: Path) -> Path:
    """Write 3000 pilot pairs and a source-linked manifest, never overwrite.

    Args:
        config_path: JSON with reference_config, sampling_seed, split_counts.
        output_dir: New directory for NPZ, manifest and copied reference bundle.

    Returns:
        Path of manifest.json. Existing destinations are rejected.
    """
    config = json.loads(config_path.read_text())
    if config["split_counts"] != [2000, 500, 500] or config["sampling_seed"] != 7:
        raise ValueError("This command is restricted to the 3000-pair seed-7 pilot")
    reference_path = PROJECT_ROOT / config["reference_config"]
    reference = load_reference_bundle(reference_path)
    start = perf_counter()
    data = build_pilot_dataset(reference, config["sampling_seed"], tuple(config["split_counts"]))
    duration = perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_dir = output_dir / "reference"
    bundle_dir.mkdir()
    for name in ("reference_config.json", "vehicle_parameters.json", "geometry_manifest.json"):
        shutil.copyfile(reference_path.parent / name, bundle_dir / name)
    archive = output_dir / "pairs.npz"
    np.savez_compressed(archive, **data)
    (output_dir / "generation_config.json").write_bytes(config_path.read_bytes())
    manifest = {
        "kind": "x4_950_pilot_pairs_v1", "purpose": "offline synthetic pilot, not final test",
        "config": config, "reference_sha256": reference["reference_sha256"],
        "mass_total_kg": reference["mass_total_kg"], "arm_pose": reference["initial_joints"],
        "reference_bundle": "reference/reference_config.json",
        "archive": archive.name, "archive_sha256": file_sha256(archive),
        "split_names": ["train", "validation", "pilot_holdout"],
        "split_counts": np.bincount(data["split"]).tolist(),
        "group_size": 10, "leakage_audit": check_split_leakage(data),
        "coordinate_units": "W ENU-like, B FLU about root O; WXYZ B→W; SI",
        "generation_seconds": duration,
        "array_memory_bytes": sum(array.nbytes for array in data.values()),
        "force_abs_max_n": float(np.max(np.abs(data["force_wind_world_n"]))),
        "label_abs_max_m_s2": float(np.max(np.abs(data["delta_acceleration_world"]))),
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                          for path in Path(__file__).parent.glob("*.py")},
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def load_dataset(manifest_path: Path) -> tuple[dict[str, NDArray], dict[str, Any]]:
    """Validate hashes/reference/leakage and load a numeric NPZ with no pickle."""
    manifest = json.loads(manifest_path.read_text())
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if content_sha256(payload) != manifest["manifest_sha256"]:
        raise ValueError("Dataset manifest fingerprint mismatch")
    archive = manifest_path.parent / manifest["archive"]
    if file_sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Dataset archive fingerprint mismatch")
    reference = load_reference_bundle(manifest_path.parent / manifest["reference_bundle"])
    if reference["reference_sha256"] != manifest["reference_sha256"]:
        raise ValueError("Dataset reference fingerprint mismatch")
    with np.load(archive, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    check_split_leakage(data)
    return data, manifest
