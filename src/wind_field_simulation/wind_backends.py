"""Runtime adapters sharing the existing force-only wind increment pipeline."""

import json
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
import torch
from numpy.typing import NDArray

from .training.reference_aerodynamics import paired_wind_labels
from .training.dataset_v1 import file_sha256
from .training.model_v1 import load_checkpoint, predict_increment
from .training.sampling import body_to_world_rotation
from .training.xml_parameters import load_reference_bundle

FloatArray = NDArray[np.float64]


class AnalyticalWindResidualModel:
    """Same-state wind increment; shares the app's existing force pipeline."""

    def __init__(
        self, reference_path: Path, xml_path: Path, model: mujoco.MjModel,
    ) -> None:
        """Validate the frozen bundle against the compiled runtime assembly.

        Args:
            reference_path: Exported, fingerprinted reference_config.json.
            xml_path: Actual scene used by the application.
            model: Already compiled runtime model, checked without mutation.
        """
        self.reference = load_reference_bundle(reference_path, xml_path)
        self.reference_sha256 = self.reference["reference_sha256"]
        self.mass_total_kg = float(self.reference["mass_total_kg"])
        # Check individual masses, not cached body_subtreemass after mutation.
        vehicle_path = reference_path.parent / "vehicle_parameters.json"
        vehicle = json.loads(vehicle_path.read_text())
        for body in vehicle["bodies"]:
            actual = model.body_mass[model.body(body["name"]).id]
            if not np.isclose(actual, body["mass_kg"], rtol=0, atol=1e-12):
                raise ValueError("Runtime model mass differs from reference")
        for name, joint in self.reference["initial_joints"].items():
            start = int(model.jnt_qposadr[model.joint(name).id])
            expected = np.asarray(joint["qpos"])
            actual = model.qpos0[start:start + expected.size]
            if not np.allclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError("Runtime model initial joints differ")
        if model.opt.density != 0 or model.opt.viscosity != 0:
            raise ValueError("Runtime fluid forces must stay disabled")
        # Validate proxy arrays and physical assumptions before connecting PX4.
        self.predict([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])

    def predict(
        self,
        velocity_world: Sequence[float],
        quaternion_wxyz: Sequence[float],
        angular_velocity_body: Sequence[float],
        wind_world: Sequence[float],
    ) -> FloatArray:
        """Return raw world-frame wind acceleration increment in m/s².

        Args:
            velocity_world: Root O velocity in world coordinates, m/s.
            quaternion_wxyz: Unit body-to-world WXYZ quaternion.
            angular_velocity_body: Body angular rate, rad/s.
            wind_world: World horizontal air velocity, resultant <=3 m/s.

        Returns:
            Complete raw same-state delta force divided by total mass.
            The caller performs XY projection, EMA and limiting. No moment,
            no no-wind drag, and no live arm geometry is additionally applied.
        """
        for name, values, size in (
            ("velocity", velocity_world, 3),
            ("quaternion", quaternion_wxyz, 4),
            ("angular rate", angular_velocity_body, 3),
            ("wind", wind_world, 3),
        ):
            if np.asarray(values).shape != (size,):
                raise ValueError(f"{name} must have shape ({size},)")
        paired = paired_wind_labels(
            velocity_world, quaternion_wxyz, angular_velocity_body,
            wind_world, self.reference,
        )
        return paired["delta_force_world_n"] / self.mass_total_kg


class X4PinnWindResidualModel:
    """Validated CPU pilot wrapper; no legacy RPM inputs or extra forces.

    Args:
        checkpoint_path: Explicit, separately preserved X4 v1 candidate.
        reference_path: Frozen reference bundle used for the candidate labels.
        xml_path: Actual scene used by the app.
        model: Compiled runtime model, checked without modifying its state.
    """

    def __init__(
        self, checkpoint_path: Path, reference_path: Path,
        xml_path: Path, model: mujoco.MjModel,
    ) -> None:
        # Reuse the same XML closure, mass, initial joints and fluid checks as
        # the analytic runtime. No reference force is added during inference.
        validated = AnalyticalWindResidualModel(reference_path, xml_path, model)
        self.network, metadata = load_checkpoint(checkpoint_path)
        vehicle = json.loads(
            (reference_path.parent / "vehicle_parameters.json").read_text())
        expected = {
            "reference_sha256": validated.reference_sha256,
            "reference_config": validated.reference,
            "mass_total_kg": validated.mass_total_kg,
            "arm_pose": validated.reference["initial_joints"],
            "xml_mesh_dependencies": vehicle["dependencies"],
            "coordinates_units": (
                "B FLU at root O; W ENU-like; WXYZ B→W; m/s, rad/s, m/s²"),
            "wind_domain": (
                "horizontal resultant 0–3 m/s; fixed initial arm geometry"),
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(f"X4 checkpoint {field} differs from runtime reference")
        for field in ("dataset_manifest_sha256", "dataset_archive_sha256"):
            digest = metadata.get(field)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError(f"X4 checkpoint has invalid {field}")
        self.reference_sha256 = validated.reference_sha256
        self.mass_total_kg = validated.mass_total_kg
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.dataset_manifest_sha256 = metadata["dataset_manifest_sha256"]
        self.dataset_archive_sha256 = metadata["dataset_archive_sha256"]
        self.inference_device = "cpu"
        self.inference_dtype = "float32"
        # Small batches are faster and reproducible without a large BLAS pool.
        # Set once in this new backend's owning process; legacy path unchanged.
        torch.set_num_threads(1)
        self.torch_threads = torch.get_num_threads()
        self.predict([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])

    def predict(
        self, velocity_world: Sequence[float], quaternion_wxyz: Sequence[float],
        angular_velocity_body: Sequence[float], wind_world: Sequence[float],
    ) -> FloatArray:
        """Return raw full W wind increment; leave XY/EMA/limit to the app.

        Args:
            velocity_world: Root O velocity [3], m/s.
            quaternion_wxyz: Unit B→W WXYZ [4].
            angular_velocity_body: Current body angular rate [3], rad/s.
            wind_world: World horizontal wind [3], resultant <=3 m/s.

        Returns:
            Raw three-axis acceleration difference [3], m/s². Zero wind is
            exactly zero. No no-wind baseline drag or torque is applied.
        """
        values = []
        for name, value, size in (
            ("velocity", velocity_world, 3), ("quaternion", quaternion_wxyz, 4),
            ("angular rate", angular_velocity_body, 3), ("wind", wind_world, 3),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape ({size},)")
            values.append(array[None, :])
        velocity, quaternion, omega, wind = values
        rotation = body_to_world_rotation(quaternion)
        return predict_increment(self.network, velocity, rotation, omega, wind,
                                  project_xy=False)[0]
