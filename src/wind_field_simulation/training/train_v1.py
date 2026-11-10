"""Small deterministic supervised/physics-constrained X4 reference pilot."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .dataset_v1 import file_sha256, load_dataset
from .model_v1 import KIND, X4WindAeroV1, fit_normalization, fit_quadratic_prior
from .reference_aerodynamics import reference_wrench_body
from .xml_parameters import PROJECT_ROOT, load_reference_bundle

DEFAULT_TRAINING = {
    "seed": 42, "epochs": 200, "batch_size": 1024, "learning_rate": 1e-3,
    "weight_decay": 1e-5, "patience": 20, "lr_patience": 5,
    "lr_factor": 0.5, "gradient_clip": 1.0, "positive_work_weight": 0.01,
    "torch_threads": 1,
}


def pair_loss(
    model: X4WindAeroV1, batch: dict[str, Tensor], full_scale: Tensor,
    delta_scale: Tensor, positive_work_weight: float,
) -> Tensor:
    """Combine full acceleration, raw XY pair and omega=0 full-work losses.

    Args:
        model: Complete B acceleration predictor.
        batch: Body air velocity pair, omega, rotation, full and delta targets.
        full_scale: Training-only full per-axis std with 0.05 floor, m/s².
        delta_scale: Independent training-only increment scale, m/s².
        positive_work_weight: Dimensionless soft constraint coefficient.

    Returns:
        Scalar dimensionless loss. Positive work applies to complete forces,
        never to wind increments. It is a soft penalty, not a guarantee.
    """
    air = batch["air"]
    count = len(air)
    omega = batch["omega"][:, None, :].expand(-1, 2, -1)
    prediction = model(air.reshape(-1, 3), omega.reshape(-1, 3)).reshape(count, 2, 3)
    delta = torch.einsum("nij,nj->ni", batch["rotation"],
                         prediction[:, 0] - prediction[:, 1])
    full_loss = ((prediction - batch["full"]) / full_scale).square().mean()
    delta_loss = ((delta[:, :2] - batch["delta"][:, :2]) / delta_scale[:2]).square().mean()
    zero_omega = torch.all(batch["omega"] == 0, dim=1)
    work_loss = prediction.new_zeros(())
    if torch.any(zero_omega):
        # Compute physical a·v BEFORE normalizing: different axis scales can
        # otherwise reverse its sign. A single positive scalar preserves sign.
        normalized_work = (prediction[zero_omega] * air[zero_omega]).sum(-1)
        normalized_work /= full_scale.mean() * (
            1 + torch.linalg.vector_norm(air[zero_omega], dim=-1))
        work_loss = torch.relu(normalized_work).square().mean()
    return full_loss + delta_loss + positive_work_weight * work_loss


def train_pilot(
    manifest_path: Path, output_dir: Path, overrides: dict[str, Any] | None = None,
) -> Path:
    """Train one bounded CPU pilot and save the best-validation checkpoint.

    Args:
        manifest_path: Fingerprinted, nominal 3000-pair pilot manifest.
        output_dir: New directory; existing runs are never overwritten.
        overrides: Explicit trial options, saved verbatim; no automatic sweeps.

    Returns:
        New candidate.pt. Holdout is not used in optimization/epoch selection.
    """
    config = {**DEFAULT_TRAINING, **(overrides or {})}
    if (not 1 <= config["epochs"] <= 200 or config["batch_size"] < 1
            or config["torch_threads"] != 1):
        raise ValueError("Pilot requires 1–200 epochs, positive batch and one CPU thread")
    data, manifest = load_dataset(manifest_path)
    if manifest["kind"] != "x4_950_pilot_pairs_v1" or len(data["split"]) != 3000:
        raise ValueError("Expected only the separate 3000-pair pilot")
    reference = load_reference_bundle(manifest_path.parent / manifest["reference_bundle"])
    vehicle = json.loads((manifest_path.parent / "reference/vehicle_parameters.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(config["seed"])
    core = np.concatenate((data["air_velocity_body_pair"],
                           np.repeat(data["omega_body"][:, None, :], 2, axis=1)), axis=2)
    stats = fit_normalization(core, data["full_acceleration_body"],
                               data["delta_acceleration_world"], data["split"] == 0)
    prior = fit_quadratic_prior(reference)
    unit_axes = np.concatenate((np.eye(3), -np.eye(3)))
    unit_acceleration = (reference_wrench_body(unit_axes, np.zeros_like(unit_axes), reference)[0]
                         / reference["mass_total_kg"])
    model = X4WindAeroV1(stats["input_mean"], stats["input_std"], stats["full_scale"], prior)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config["lr_factor"], patience=config["lr_patience"])
    tensors = {name: torch.tensor(data[key], dtype=torch.float32)
               for name, key in (("air", "air_velocity_body_pair"), ("omega", "omega_body"),
                                 ("rotation", "rotation_body_to_world"),
                                 ("full", "full_acceleration_body"),
                                 ("delta", "delta_acceleration_world"))}
    train_ids = torch.tensor(np.flatnonzero(data["split"] == 0))
    validation = {key: value[data["split"] == 1] for key, value in tensors.items()}
    full_scale = torch.tensor(stats["full_scale"], dtype=torch.float32)
    delta_scale = torch.tensor(stats["delta_scale"], dtype=torch.float32)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state: dict[str, Tensor] = {}
    history = []
    start = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        ordered_ids = train_ids[torch.randperm(len(train_ids), generator=generator)]
        total = 0.
        for ids in ordered_ids.split(config["batch_size"]):
            batch = {key: value[ids] for key, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = pair_loss(model, batch, full_scale, delta_scale, config["positive_work_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(pair_loss(model, validation, full_scale, delta_scale,
                                               config["positive_work_weight"]))
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        row = {"epoch": epoch, "training_loss": total/len(train_ids),
               "validation_loss": validation_loss, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if stale >= config["patience"]:
            break
    elapsed = perf_counter() - start
    saved = {
        "kind": KIND, "feature_version": "body_air_velocity_omega_6d_v1",
        "architecture": "fixed nonnegative quadratic prior + 6-64-64-3 Tanh residual",
        "state_dict": best_state, "normalization": stats, "prior_coefficients": prior.tolist(),
        "prior_source": "nominal ±unit-axis least squares; not measured",
        "prior_fit_samples": {"air_velocity_body_m_s": unit_axes.tolist(),
                              "omega_body_rad_s": np.zeros_like(unit_axes).tolist(),
                              "full_acceleration_body_m_s2": unit_acceleration.tolist()},
        "reference_sha256": manifest["reference_sha256"],
        "reference_config": reference,
        "xml_mesh_dependencies": vehicle["dependencies"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "dataset_archive_sha256": manifest["archive_sha256"],
        "mass_total_kg": reference["mass_total_kg"], "arm_pose": reference["initial_joints"],
        "coordinates_units": "B FLU at root O; W ENU-like; WXYZ B→W; m/s, rad/s, m/s²",
        "wind_domain": "horizontal resultant 0–3 m/s; fixed initial arm geometry",
        "training_config": config, "sampling_seed": manifest["config"]["sampling_seed"],
        "best_validation_loss": best_loss, "best_epoch": best_epoch,
        "epochs_executed": len(history), "training_seconds": elapsed,
        "device": "cpu", "dtype": "float32", "deterministic_algorithms": True,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "platform": platform.platform()},
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                          for path in Path(__file__).parent.glob("*.py")},
        "status": "pilot only; no runtime switch or true aerodynamic accuracy claim",
    }
    path = output_dir / "candidate.pt"
    torch.save(saved, path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    summary = {key: value for key, value in saved.items() if key not in ("state_dict", "reference_config")}
    summary["checkpoint_sha256"] = file_sha256(path)
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checkpoint": str(path), "best_epoch": best_epoch,
                      "best_validation_loss": best_loss, "seconds": elapsed}), flush=True)
    return path
