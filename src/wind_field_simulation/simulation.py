#!/usr/bin/env python3
"""Standalone PX4/MuJoCo wind-field experiment.

The module is intentionally importable so its numerical models and control
state can be verified without opening MAVLink sockets or graphical windows.

Examples:
    Start unconfigured; use Apply Wind before enabling wind::

        python -m wind_field_simulation

    Prefill a draft; Apply Wind is still required::

        python -m wind_field_simulation \
            --wind-x 1 --wind-y 0
"""

from __future__ import annotations

import argparse
import enum
import faulthandler
import hashlib
import os
import queue
import sys
import textwrap
import threading
import time
from types import TracebackType
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, TextBox

from .sitl_process import request_managed_sitl_shutdown
from .run_logging import WindRunLogger
from .wind_backends import AnalyticalWindResidualModel, X4PinnWindResidualModel


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

GRAVITY_M_S2 = 9.80665
ROTOR_THRUST_ACTUATOR_NAMES = ("rotor0", "rotor1", "rotor2", "rotor3")
ROTOR_SPIN_ACTUATOR_NAMES = (
    "rotor_a00_spin",
    "rotor_a10_spin",
    "rotor_a20_spin",
    "rotor_a30_spin",
)
WIND_ARROW_SHAFT_RGBA = np.array(
    [0.25, 0.55, 1.0, 0.50],
    dtype=np.float32,
)
WIND_ARROW_TIP_RGBA = np.array(
    [0.25, 0.55, 1.0, 0.65],
    dtype=np.float32,
)
WIND_ARROW_MIN_SPEED_M_S = 0.1
WIND_ARROW_LENGTH_M = 0.28
WIND_ARROW_HEAD_LENGTH_M = 0.035
WIND_ARROW_FLOW_PERIOD_M = 2.5
TRAJECTORY_PLOT_UPDATE_PERIOD_S = 0.25
TRAJECTORY_PLOT_TARGET_SAMPLE_COUNT = 240
BAROMETER_RANDOM_SEED = 0
BAROMETER_NOISE_STD_HPA = 0.01  # 1 Pa, approximately 8 cm at sea level.


def repository_root() -> Path:
    """Returns the development project root (resources use the package path)."""
    return Path(__file__).resolve().parents[2]


def default_pinn_project_root() -> Path:
    """Returns the bundled PINN resource root, independent of the old checkout."""
    return Path(__file__).resolve().parent / "assets"


def default_pinn_checkpoint_dir() -> Path:
    """Returns the bundled checkpoint and normalization directory."""
    return default_pinn_project_root() / "checkpoints"


def default_x4_pinn_checkpoint() -> Path:
    """Return the separately bundled, verified X4 v1 pilot checkpoint."""
    return default_pinn_checkpoint_dir() / "x4_950_wind_v1" / "candidate.pt"


def default_wind_reference_config() -> Path:
    """Return the frozen bundle shared by default X4 PINN and analytic fallback."""
    return default_pinn_checkpoint_dir() / "x4_950_wind_v1" / "reference_config.json"


def default_xml_path() -> Path:
    """Returns the bundled quadrotor-with-arm scene."""
    return default_pinn_project_root() / "scenes" / "default" / "robot_quadrotor_950_with_arm.xml"


def default_params_path() -> Path:
    """Returns the bundled X4-950 parameter export."""
    return default_pinn_project_root() / "configs" / "X4-950.params"


from ._vendor.pinn import ResidualPINN  # noqa: E402
from ._vendor.dataset import Normalizer  # noqa: E402
from ._vendor.hummingbird_params import quad_params  # noqa: E402
from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import ardupilotmega as mavlink2  # noqa: E402


PINN_EMA_ALPHA = 0.05
PINN_ACCELERATION_LIMIT_M_S2 = 6.0
# The PINN computes a float32 norm and quadratic drag. Reserve 1/eps
# headroom in squared wind speed, rather than accepting all finite float64s.
# This is a numerical envelope, not a validated physical/training range.
PINN_WIND_NUMERICAL_LIMIT_M_S = float(np.sqrt(
    float(np.finfo(np.float32).max) * float(np.finfo(np.float32).eps)
))
HOVER_OMEGA_RAD_S = float(
    np.sqrt(
        float(quad_params["mass"])
        * 9.81
        / (4.0 * float(quad_params["k_eta"]))
    )
)


def _validated_vector3(values: Sequence[float], name: str) -> FloatArray:
    """Returns a finite three-element floating-point vector.

    Args:
        values: Candidate vector values.
        name: Field name used in validation errors.

    Returns:
        A defensive NumPy copy with shape ``(3,)``.

    Raises:
        ValueError: If the vector has the wrong shape or non-finite values.
    """
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector.copy()


def _validated_wind_velocity(values: Sequence[float | str]) -> FloatArray:
    """Validate horizontal wind with resultant speed at most 3 m/s.

    Args:
        values: World-frame [X, Y, 0] wind velocity in m/s.

    Returns:
        An owned float64 velocity vector safe for downstream float32 scaling.

    Raises:
        ValueError: If values are invalid, nonhorizontal or numerically unsafe.
    """
    try:
        velocity = _validated_vector3(values, "wind_velocity_world")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Enter finite X/Y values.") from error
    if velocity[2] != 0.0:
        raise ValueError("Only horizontal wind is supported (Z=0).")
    scaled_speed = float(np.hypot.reduce(
        velocity / PINN_WIND_NUMERICAL_LIMIT_M_S,
    ))
    if scaled_speed > 1.0:
        raise ValueError("Wind exceeds PINN numeric range.")
    if np.hypot(velocity[0], velocity[1]) > np.nextafter(3.0, np.inf):
        raise ValueError("Horizontal wind resultant must be <=3 m/s.")
    return velocity


class PinnWindResidualModel:
    """Loads and evaluates the checkpoint-compatible residual PINN."""

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Loads model weights and normalization statistics.

        Args:
            checkpoint_dir: Directory containing both checkpoint files.
            device: Explicit inference device or automatic CUDA/CPU selection.

        Raises:
            FileNotFoundError: If either checkpoint file is absent.
            ValueError: If checkpoint metadata or normalization is invalid.
            RuntimeError: If the state dictionary does not match the model.
        """
        directory = Path(
            checkpoint_dir or default_pinn_checkpoint_dir()
        ).resolve()
        model_path = directory / "best_model.pt"
        normalizer_path = directory / "normalizer.pt"
        for required_path in (model_path, normalizer_path):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"PINN checkpoint not found: {required_path}"
                )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError(
                f"PINN checkpoint lacks model_state: {model_path}"
            )
        self.model = ResidualPINN(input_dim=17).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        normalizer = Normalizer().load(str(normalizer_path))
        self.mean = normalizer.mean.to(
            device=self.device,
            dtype=torch.float32,
        )
        self.std = normalizer.std.to(
            device=self.device,
            dtype=torch.float32,
        )
        if self.mean.shape != (17,) or self.std.shape != (17,):
            raise ValueError("PINN normalizer must contain 17 features")
        if not bool(torch.isfinite(self.mean).all().item()):
            raise ValueError("PINN normalizer mean must be finite")
        if not bool(torch.isfinite(self.std).all().item()):
            raise ValueError("PINN normalizer std must be finite")
        if not bool((self.std > 0.0).all().item()):
            raise ValueError("PINN normalizer std must be positive")

    @staticmethod
    def build_features(
        velocity_world: Sequence[float],
        quaternion_wxyz: Sequence[float],
        angular_velocity_body: Sequence[float],
        wind_world: Sequence[float],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Builds the checkpoint's 17-dimensional physical feature vector.

        Args:
            velocity_world: MuJoCo world-frame velocity in m/s.
            quaternion_wxyz: MuJoCo body-to-world quaternion in WXYZ order.
            angular_velocity_body: Body-frame angular velocity in rad/s.
            wind_world: Active MuJoCo world-frame wind in m/s.

        Returns:
            Float32 feature vector and physical relative velocity.

        Raises:
            ValueError: If any input has an invalid shape or value.
        """
        velocity = _validated_vector3(
            velocity_world,
            "velocity_world",
        ).astype(np.float32)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError(
                "quaternion_wxyz must be finite with shape (4,)"
            )
        angular_velocity = _validated_vector3(
            angular_velocity_body,
            "angular_velocity_body",
        ).astype(np.float32)
        wind = _validated_vector3(
            wind_world,
            "wind_world",
        ).astype(np.float32)
        relative_velocity = velocity - wind
        features = np.concatenate(
            [
                relative_velocity,
                quaternion[[1, 2, 3, 0]].astype(np.float32),
                angular_velocity,
                np.full(4, HOVER_OMEGA_RAD_S, dtype=np.float32),
                wind,
            ]
        )
        return features, relative_velocity

    def predict(
        self,
        velocity_world: Sequence[float],
        quaternion_wxyz: Sequence[float],
        angular_velocity_body: Sequence[float],
        wind_world: Sequence[float],
    ) -> FloatArray:
        """Predicts finite world-frame residual acceleration.

        Args:
            velocity_world: MuJoCo world-frame velocity in m/s.
            quaternion_wxyz: MuJoCo body-to-world quaternion in WXYZ order.
            angular_velocity_body: Body-frame angular velocity in rad/s.
            wind_world: Active MuJoCo world-frame wind in m/s.

        Returns:
            Three-axis world-frame residual acceleration in m/s².

        Raises:
            ValueError: If any state input is invalid.
            FloatingPointError: If the model result is invalid.
        """
        features, relative_velocity = self.build_features(
            velocity_world,
            quaternion_wxyz,
            angular_velocity_body,
            wind_world,
        )
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        relative_velocity_tensor = torch.as_tensor(
            relative_velocity,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        normalized = (feature_tensor - self.mean) / self.std
        with torch.inference_mode():
            prediction = self.model(
                normalized,
                relative_velocity_tensor,
            )
        result = (
            prediction.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "PINN prediction must be finite with shape (3,)"
            )
        return result


@dataclass
class ResidualSafetyFilter:
    """Applies stateful EMA smoothing and per-axis acceleration limits."""

    ema_alpha: float = PINN_EMA_ALPHA
    acceleration_limit_m_s2: float = PINN_ACCELERATION_LIMIT_M_S2
    filtered: FloatArray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
        init=False,
    )

    def update(self, prediction: Sequence[float]) -> FloatArray:
        """Updates and returns the limited residual acceleration.

        Args:
            prediction: Raw three-axis residual acceleration in m/s².

        Returns:
            EMA-filtered and per-axis limited acceleration.

        Raises:
            ValueError: If parameters or prediction are invalid.
        """
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if (
            not np.isfinite(self.acceleration_limit_m_s2)
            or self.acceleration_limit_m_s2 <= 0.0
        ):
            raise ValueError(
                "acceleration_limit_m_s2 must be finite and positive"
            )
        residual = _validated_vector3(prediction, "prediction")
        self.filtered = (
            self.ema_alpha * residual
            + (1.0 - self.ema_alpha) * self.filtered
        )
        return np.clip(
            self.filtered,
            -self.acceleration_limit_m_s2,
            self.acceleration_limit_m_s2,
        ).copy()

    def reset(self) -> None:
        """Clears the stored EMA state."""
        self.filtered = np.zeros(3, dtype=np.float64)


def _add_scene_triangle_tip(
    scene: Any,
    tip: FloatArray,
    direction: FloatArray,
    rgba: Sequence[float],
) -> bool:
    """Add a filled arrowhead lying entirely in the world XY plane.

    Args:
        scene: Writable MuJoCo user scene.
        tip: World-frame arrow tip position in metres.
        direction: Unit horizontal world-frame direction.
        rgba: Four-component triangle color.

    Returns:
        ``True`` when the triangle was added, otherwise ``False``.
    """
    if scene.ngeom >= scene.maxgeom:
        return False
    lateral = np.array([-direction[1], direction[0], 0.0])
    # MuJoCo TRIANGLE vertices are (0,0,0), (size[0],0,0), (0,size[1],0).
    # Orient equal orthogonal legs backward from the tip, symmetrically about
    # the wind direction. The local normal stays aligned with world +Z.
    # Source: MuJoCo 3.6 src/render/classic/render_gl3.c, mjGEOM_TRIANGLE.
    rotation = np.column_stack((
        (-direction + lateral) / np.sqrt(2.0),
        (-direction - lateral) / np.sqrt(2.0),
        np.array([0.0, 0.0, 1.0]),
    ))
    leg_length = np.sqrt(2.0) * WIND_ARROW_HEAD_LENGTH_M
    geometry = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geometry,
        mujoco.mjtGeom.mjGEOM_TRIANGLE,
        np.array([leg_length, leg_length, 0.0], dtype=np.float64),
        tip,
        rotation.ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def _add_scene_capsule(
    scene: Any,
    start: Sequence[float],
    end: Sequence[float],
    width: float,
    rgba: Sequence[float],
) -> bool:
    """Adds one capsule connector to a MuJoCo user scene.

    Args:
        scene: Writable MuJoCo user scene.
        start: Capsule start point in MuJoCo world coordinates.
        end: Capsule end point in MuJoCo world coordinates.
        width: Capsule width in metres.
        rgba: Four-component capsule color.

    Returns:
        ``True`` when the capsule was added, otherwise ``False``.
    """
    if scene.ngeom >= scene.maxgeom:
        return False
    geometry = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geometry,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geometry,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(width),
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


def draw_wind_arrows(
    scene: Any,
    wind_world: Sequence[float],
    simulation_time_s: float,
) -> None:
    """Rebuild horizontal blue shafts and filled triangular arrowheads.

    Args:
        scene: Writable user scene owned by the passive MuJoCo viewer.
        wind_world: Active world wind; visualization uses its XY projection.
        simulation_time_s: Current MuJoCo simulation time in seconds.
    """
    wind = _validated_vector3(wind_world, "wind_world")
    wind[2] = 0.0
    scene.ngeom = 0
    wind_speed = float(np.linalg.norm(wind))
    if wind_speed <= WIND_ARROW_MIN_SPEED_M_S:
        return

    direction = wind / wind_speed
    arrow_displacement = direction * WIND_ARROW_LENGTH_M
    flow_offset = (
        float(simulation_time_s) * wind_speed
    ) % WIND_ARROW_FLOW_PERIOD_M
    for x_position in np.linspace(-10.0, 10.0, 5):
        for y_position in np.linspace(-8.0, 8.0, 8):
            for z_position in np.linspace(0.0, 4.0, 3):
                # Keep shaft and head together even with a tiny user scene.
                if scene.maxgeom - scene.ngeom < 2:
                    return
                origin = np.array(
                    [x_position, y_position, z_position],
                    dtype=np.float64,
                ) + flow_offset * direction
                tip = origin + arrow_displacement
                _add_scene_capsule(
                    scene,
                    origin,
                    tip - WIND_ARROW_HEAD_LENGTH_M * direction,
                    0.009,
                    WIND_ARROW_SHAFT_RGBA,
                )
                _add_scene_triangle_tip(
                    scene,
                    tip,
                    direction,
                    WIND_ARROW_TIP_RGBA,
                )


def read_px4_param(path: Path, name: str, default: float) -> float:
    """Reads a numeric value from a PX4 parameter export.

    Args:
        path: PX4 ``.params`` file.
        name: Parameter name to locate.
        default: Value returned when the file or value is unavailable.

    Returns:
        Parsed parameter value or ``default``.
    """
    if not path.exists():
        return float(default)
    with path.open("r", encoding="utf-8", errors="replace") as param_file:
        for line in param_file:
            fields = line.strip().split()
            if len(fields) >= 5 and fields[2] == name:
                try:
                    return float(fields[3])
                except ValueError:
                    return float(default)
    return float(default)


def compute_hover_matched_total_max_thrust(
    mass_kg: float,
    hover_thrust: float,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> float:
    """Computes full-scale thrust from PX4 normalized hover thrust.

    Args:
        mass_kg: Vehicle subtree mass in kilograms.
        hover_thrust: PX4 ``MPC_THR_HOVER`` normalized command.
        gravity_m_s2: Gravity magnitude in m/s^2.

    Returns:
        Total four-rotor thrust at normalized command one, in newtons.
    """
    values = np.array([mass_kg, hover_thrust, gravity_m_s2], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("mass, hover thrust, and gravity must be finite and positive")
    return float(mass_kg * gravity_m_s2 / hover_thrust)


def equal_rotor_thrust_scales(
    total_max_thrust: float,
    rotor_count: int = 4,
) -> FloatArray:
    """Returns equal normalized-output-to-thrust scales for all rotors."""
    if rotor_count <= 0:
        raise ValueError("rotor_count must be positive")
    scale = float(total_max_thrust) / float(rotor_count)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("total_max_thrust must be finite and positive")
    return np.full(rotor_count, scale, dtype=np.float64)


def motor_outputs_to_thrusts(
    normalized_outputs: Sequence[float],
    rotor_thrust_scales: Sequence[float],
) -> FloatArray:
    """Maps four PX4 normalized allocator outputs to thrust in newtons."""
    outputs = np.asarray(normalized_outputs, dtype=np.float64)
    scales = np.asarray(rotor_thrust_scales, dtype=np.float64)
    if outputs.shape != (4,):
        raise ValueError("normalized_outputs must have shape (4,)")
    if scales.shape != (4,):
        raise ValueError("rotor_thrust_scales must have shape (4,)")
    if not np.all(np.isfinite(outputs)) or not np.all(np.isfinite(scales)):
        raise ValueError("motor outputs and thrust scales must be finite")
    if np.any(scales <= 0.0):
        raise ValueError("rotor_thrust_scales must be positive")
    return np.clip(outputs, 0.0, 1.0) * scales


@dataclass(frozen=True)
class ModelBindings:
    """Resolved MuJoCo IDs and addresses required by the simulation."""

    quadrotor_body_id: int
    main_joint_id: int
    main_qpos_adr: int
    main_dof_adr: int
    rotor_thrust_actuator_ids: IntArray
    rotor_spin_actuator_ids: IntArray
    uav_mass_kg: float

    @classmethod
    def resolve(cls, model: mujoco.MjModel) -> "ModelBindings":
        """Resolves every required model element by name.

        Args:
            model: Loaded MuJoCo model.

        Returns:
            Validated named bindings.

        Raises:
            ValueError: If any required body, joint, or actuator is missing.
        """
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "quadrotor_950_pos",
        )
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "main_joint",
        )
        thrust_ids = np.array(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name,
                )
                for name in ROTOR_THRUST_ACTUATOR_NAMES
            ],
            dtype=int,
        )
        spin_ids = np.array(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name,
                )
                for name in ROTOR_SPIN_ACTUATOR_NAMES
            ],
            dtype=int,
        )
        missing: list[str] = []
        if body_id < 0:
            missing.append("body:quadrotor_950_pos")
        if joint_id < 0:
            missing.append("joint:main_joint")
        missing.extend(
            f"actuator:{name}"
            for name, actuator_id in zip(
                ROTOR_THRUST_ACTUATOR_NAMES,
                thrust_ids,
            )
            if actuator_id < 0
        )
        missing.extend(
            f"actuator:{name}"
            for name, actuator_id in zip(ROTOR_SPIN_ACTUATOR_NAMES, spin_ids)
            if actuator_id < 0
        )
        if missing:
            raise ValueError("missing required MuJoCo elements: " + ", ".join(missing))

        uav_mass = float(model.body_subtreemass[body_id])
        if not np.isfinite(uav_mass) or uav_mass <= 0.0:
            raise ValueError("quadrotor_950_pos subtree mass must be positive")
        return cls(
            quadrotor_body_id=int(body_id),
            main_joint_id=int(joint_id),
            main_qpos_adr=int(model.jnt_qposadr[joint_id]),
            main_dof_adr=int(model.jnt_dofadr[joint_id]),
            rotor_thrust_actuator_ids=thrust_ids,
            rotor_spin_actuator_ids=spin_ids,
            uav_mass_kg=uav_mass,
        )


def standard_atmosphere_pressure_hpa(altitude_m: float) -> float:
    """Returns pressure using PX4's first-layer standard atmosphere model.

    Formula and constants follow ``src/lib/atmosphere/atmosphere.cpp`` and
    ``atmosphere.h`` (US Standard Atmosphere 1976, up to 11 km).

    Args:
        altitude_m: Altitude above mean sea level in metres.

    Returns:
        Ideal absolute pressure in hPa, as required by HIL_SENSOR.

    Raises:
        ValueError: If altitude is non-finite or outside the first layer.
    """
    if not np.isfinite(altitude_m) or altitude_m > 11_000.0:
        raise ValueError("barometer altitude must be finite and <= 11000 m")
    temperature_ratio = (288.15 - 0.0065 * altitude_m) / 288.15
    return float(1013.25 * temperature_ratio ** (GRAVITY_M_S2 / (0.0065 * 287.1)))


class Px4Bridge:
    """Independent MAVLink bridge between MuJoCo and PX4 SITL."""

    EARTH_RADIUS_M = 6_378_137.0
    HOME_LAT_DEG = 22.0
    HOME_LON_DEG = 114.0
    HOME_ALT_M = 100.0
    PX4_OFFBOARD_MAIN_MODE = 6
    POSITION_ONLY_TARGET_TYPE_MASK = 0b0000101111111000
    MUJOCO_TO_PX4_AXIS = np.array([1.0, -1.0, -1.0], dtype=np.float64)

    def __init__(
        self,
        rotor_thrust_scales: Sequence[float],
        sim_tcp_port: int = 4560,
        command_udp_port: int = 14580,
        connection_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Initializes PX4 endpoints and conversion parameters.

        Args:
            rotor_thrust_scales: Four normalized-output-to-thrust scales in N.
            sim_tcp_port: PX4 simulator MAVLink TCP port.
            command_udp_port: PX4 Offboard MAVLink UDP port.
            connection_factory: Injectable pymavlink connection constructor.
            clock: Injectable monotonic wall clock.
        """
        scales = np.asarray(rotor_thrust_scales, dtype=np.float64)
        if scales.shape != (4,) or not np.all(np.isfinite(scales)):
            raise ValueError("rotor_thrust_scales must be finite with shape (4,)")
        if np.any(scales <= 0.0):
            raise ValueError("rotor_thrust_scales must be positive")
        self.rotor_thrust_scales = scales.copy()
        self.sim_tcp_port = int(sim_tcp_port)
        self.command_udp_port = int(command_udp_port)
        self._connection_factory = (
            mavutil.mavlink_connection
            if connection_factory is None
            else connection_factory
        )
        self._clock = clock
        self.sim_link: Any | None = None
        self.command_link: Any | None = None
        self.command_target_system = 1
        self.command_target_component = 1
        self.local_origin_world: FloatArray | None = None
        self.last_rotor_thrusts = np.zeros(4, dtype=np.float64)
        self.last_control_received_wall_time = 0.0
        self.last_heartbeat_wall_time = 0.0
        self.last_command_heartbeat_wall_time = 0.0
        self.last_px4_heartbeat_wall_time = 0.0
        self.last_offboard_request_wall_time = 0.0
        self.last_arm_request_wall_time: float | None = None
        self.request_interval_s = 0.25
        self.armed = False
        self.offboard = False
        self.shutdown_request_sent = False
        self.last_command_ack = "none"
        self.arm_ack_sequence = 0
        self.last_arm_ack_result: int | None = None
        self.last_statustext = "none"
        self._barometer_rng = np.random.default_rng(BAROMETER_RANDOM_SEED)
        self._statustext_chunks: dict[
            tuple[int, int, int], tuple[int, bytes, float]
        ] = {}

    @property
    def connected(self) -> bool:
        """Reports whether both MAVLink connection objects exist."""
        return self.sim_link is not None and self.command_link is not None

    def connect(self) -> None:
        """Opens simulator and Offboard MAVLink connections."""
        self.sim_link = self._connection_factory(
            f"tcpin:127.0.0.1:{self.sim_tcp_port}",
            source_system=2,
            source_component=1,
            autoreconnect=True,
        )
        self.command_link = self._connection_factory(
            f"udpout:127.0.0.1:{self.command_udp_port}",
            source_system=206,
            source_component=190,
        )
        # MAVLink 1 drops STATUSTEXT.id/chunk_seq extension fields. Use a
        # per-link MAVLink 2 codec without changing mavutil's global dialect.
        self.command_link.mav = mavlink2.MAVLink(
            self.command_link, srcSystem=206, srcComponent=190
        )
        self.command_link.mav.robust_parsing = True
        self.command_link.WIRE_PROTOCOL_VERSION = "2.0"
        self.command_link.first_byte = False

    def close(self) -> None:
        """Closes both MAVLink connections when supported."""
        for link in (self.sim_link, self.command_link):
            if link is not None and hasattr(link, "close"):
                link.close()
        self.sim_link = None
        self.command_link = None

    @classmethod
    def mujoco_world_to_px4_local(
        cls,
        vector_world: Sequence[float],
    ) -> FloatArray:
        """Converts a MuJoCo ENU-like world vector to PX4 NED."""
        return _validated_vector3(vector_world, "vector_world") * cls.MUJOCO_TO_PX4_AXIS

    @classmethod
    def mujoco_body_to_px4_body(
        cls,
        vector_body: Sequence[float],
    ) -> FloatArray:
        """Converts a MuJoCo FLU body vector to PX4 FRD."""
        return _validated_vector3(vector_body, "vector_body") * cls.MUJOCO_TO_PX4_AXIS

    def set_local_origin(self, position_world: Sequence[float]) -> None:
        """Sets the MuJoCo world position corresponding to PX4 local zero."""
        self.local_origin_world = _validated_vector3(
            position_world,
            "position_world",
        )

    def world_position_to_px4_local(
        self,
        position_world: Sequence[float],
    ) -> FloatArray:
        """Converts an absolute MuJoCo position to PX4 local NED."""
        position = _validated_vector3(position_world, "position_world")
        if self.local_origin_world is None:
            self.set_local_origin(position)
        assert self.local_origin_world is not None
        return self.mujoco_world_to_px4_local(position - self.local_origin_world)

    @staticmethod
    def _quat_wxyz_to_matrix(quaternion_wxyz: Sequence[float]) -> FloatArray:
        """Converts a normalized WXYZ quaternion to a rotation matrix."""
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion_wxyz must be finite with shape (4,)")
        quaternion /= max(float(np.linalg.norm(quaternion)), np.finfo(float).eps)
        w_value, x_value, y_value, z_value = quaternion
        return np.array(
            [
                [
                    1.0 - 2.0 * (y_value**2 + z_value**2),
                    2.0 * (x_value * y_value - z_value * w_value),
                    2.0 * (x_value * z_value + y_value * w_value),
                ],
                [
                    2.0 * (x_value * y_value + z_value * w_value),
                    1.0 - 2.0 * (x_value**2 + z_value**2),
                    2.0 * (y_value * z_value - x_value * w_value),
                ],
                [
                    2.0 * (x_value * z_value - y_value * w_value),
                    2.0 * (y_value * z_value + x_value * w_value),
                    1.0 - 2.0 * (x_value**2 + y_value**2),
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_to_quat_wxyz(rotation_matrix: FloatArray) -> FloatArray:
        """Converts a three-dimensional rotation matrix to WXYZ quaternion."""
        matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("rotation_matrix must be finite with shape (3, 3)")
        quaternion = np.empty(4, dtype=np.float64)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion[:] = [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        else:
            axis = int(np.argmax(np.diag(matrix)))
            next_axis = (axis + 1) % 3
            last_axis = (axis + 2) % 3
            scale = 2.0 * np.sqrt(
                max(
                    0.0,
                    1.0
                    + matrix[axis, axis]
                    - matrix[next_axis, next_axis]
                    - matrix[last_axis, last_axis],
                )
            )
            scale = max(scale, np.finfo(float).eps)
            quaternion[axis + 1] = 0.25 * scale
            quaternion[0] = (
                matrix[last_axis, next_axis] - matrix[next_axis, last_axis]
            ) / scale
            quaternion[next_axis + 1] = (
                matrix[axis, next_axis] + matrix[next_axis, axis]
            ) / scale
            quaternion[last_axis + 1] = (
                matrix[axis, last_axis] + matrix[last_axis, axis]
            ) / scale
        return quaternion / max(
            float(np.linalg.norm(quaternion)),
            np.finfo(float).eps,
        )

    @classmethod
    def mujoco_quat_to_px4_quat(
        cls,
        quaternion_wxyz: Sequence[float],
    ) -> FloatArray:
        """Converts MuJoCo world/body attitude to PX4 NED/FRD attitude."""
        axis_flip = np.diag(cls.MUJOCO_TO_PX4_AXIS)
        rotation_mujoco = cls._quat_wxyz_to_matrix(quaternion_wxyz)
        return cls._matrix_to_quat_wxyz(
            axis_flip @ rotation_mujoco @ axis_flip
        )

    def request_arm(self, force: bool = False) -> None:
        """Sends a rate-limited PX4 Arm command and no mode command."""
        if self.command_link is None:
            return
        wall_time = self._clock()
        if (
            not force
            and self.last_arm_request_wall_time is not None
            and wall_time - self.last_arm_request_wall_time
            < self.request_interval_s
        ):
            return
        self.last_arm_request_wall_time = wall_time
        self.command_link.mav.command_long_send(
            self.command_target_system,
            self.command_target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0,
            21196.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def request_disarm(self) -> None:
        """Sends only a PX4 Disarm command."""
        if self.command_link is None:
            return
        self.command_link.mav.command_long_send(
            self.command_target_system,
            self.command_target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0.0,
            21196.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def request_shutdown(self) -> None:
        """Sends one PX4 autopilot shutdown command when connected."""
        if self.command_link is None or self.shutdown_request_sent:
            return
        self.command_link.mav.command_long_send(
            self.command_target_system,
            self.command_target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.shutdown_request_sent = True

    def request_offboard(self, force: bool = False) -> None:
        """Sends only a PX4 Offboard mode request."""
        if self.command_link is None:
            return
        wall_time = self._clock()
        if (
            not force
            and self.last_offboard_request_wall_time > 0.0
            and wall_time - self.last_offboard_request_wall_time
            < self.request_interval_s
        ):
            return
        self.last_offboard_request_wall_time = wall_time
        self.command_link.mav.command_long_send(
            self.command_target_system,
            self.command_target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            29.0,
            float(self.PX4_OFFBOARD_MAIN_MODE),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def send_simulator_heartbeat(self) -> None:
        """Sends the one-hertz simulator heartbeat."""
        if self.sim_link is None:
            return
        wall_time = self._clock()
        if wall_time - self.last_heartbeat_wall_time < 1.0:
            return
        self.last_heartbeat_wall_time = wall_time
        self.sim_link.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GENERIC,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def send_command_heartbeat(self) -> None:
        """Identifies the interactive control panel as a GCS at one hertz.

        PX4's streams/STATUSTEXT.hpp gates text on is_gcs_connected().
        An onboard-controller heartbeat alone therefore hides arm failures.
        """
        if self.command_link is None:
            return
        wall_time = self._clock()
        if wall_time - self.last_command_heartbeat_wall_time < 1.0:
            return
        self.last_command_heartbeat_wall_time = wall_time
        self.command_link.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def send_state(
        self,
        position_world: Sequence[float],
        velocity_world: Sequence[float],
        acceleration_world: Sequence[float],
        quaternion_wxyz: Sequence[float],
        angular_rate_body: Sequence[float],
        specific_force_body: Sequence[float],
    ) -> None:
        """Sends sensors and paired float position/HIL state to PX4 SITL.

        The custom simulator consumes LOCAL_POSITION_NED immediately before
        HIL_STATE_QUATERNION when PX4_SIM_FLOAT_STATE=1. Both packets must
        share the sender, timestamp and setpoint origin. Legacy mode ignores
        the additional local-position packet.
        """
        if self.sim_link is None:
            return
        self.send_simulator_heartbeat()
        now_us = int(self._clock() * 1_000_000)
        position = _validated_vector3(position_world, "position_world")
        position_px4 = self.world_position_to_px4_local(position)
        velocity_px4 = self.mujoco_world_to_px4_local(velocity_world)
        acceleration_px4 = self.mujoco_world_to_px4_local(acceleration_world)
        quaternion_px4 = self.mujoco_quat_to_px4_quat(quaternion_wxyz)
        angular_rate_px4 = self.mujoco_body_to_px4_body(angular_rate_body)
        specific_force_px4 = self.mujoco_body_to_px4_body(specific_force_body)
        latitude = self.HOME_LAT_DEG + np.degrees(
            position_px4[0] / self.EARTH_RADIUS_M
        )
        longitude = self.HOME_LON_DEG + np.degrees(
            position_px4[1]
            / (
                self.EARTH_RADIUS_M
                * np.cos(np.radians(self.HOME_LAT_DEG))
            )
        )
        altitude = self.HOME_ALT_M - position_px4[2]
        # A constant, noiseless pressure trips DataValidator's STALE flag
        # despite fresh timestamps. Simulate a seeded measurement instead.
        pressure_hpa = standard_atmosphere_pressure_hpa(float(altitude))
        pressure_hpa += float(
            self._barometer_rng.normal(0.0, BAROMETER_NOISE_STD_HPA)
        )
        self.sim_link.mav.hil_sensor_send(
            now_us,
            *specific_force_px4.tolist(),
            *angular_rate_px4.tolist(),
            0.2,
            0.0,
            0.45,
            pressure_hpa,
            0.0,
            float(altitude),
            15.0 - 0.0065 * float(altitude),
            0x1FFF,
            0,
        )
        self.sim_link.mav.local_position_ned_send(
            (now_us // 1000) & 0xFFFFFFFF,
            *position_px4.tolist(),
            *velocity_px4.tolist(),
        )
        # HIL_STATE_QUATERNION has signed int16 velocity/acceleration fields.
        # Ground-contact impulses can exceed their range. Saturate only this
        # integer representation; float LOCAL_POSITION_NED and HIL_SENSOR
        # above retain their full values. Keep the existing receiver's /1000
        # acceleration convention (SimulatorMavlink.cpp), not a unit redesign.
        velocity_integer = np.clip(
            velocity_px4 * 100, -32768, 32767,
        ).astype(np.int16)
        acceleration_integer = np.clip(
            acceleration_px4 * 1000, -32768, 32767,
        ).astype(np.int16)
        self.sim_link.mav.hil_state_quaternion_send(
            now_us,
            quaternion_px4.tolist(),
            *angular_rate_px4.tolist(),
            int(latitude * 1e7),
            int(longitude * 1e7),
            int(altitude * 1000),
            *velocity_integer.tolist(),
            0,
            0,
            *acceleration_integer.tolist(),
        )

    def send_position_target(
        self,
        target_world: Sequence[float],
        target_yaw_rad: float,
    ) -> None:
        """Sends a position-only PX4 local-NED Offboard setpoint."""
        if self.command_link is None:
            return
        target_px4 = self.world_position_to_px4_local(target_world)
        self.command_link.mav.set_position_target_local_ned_send(
            int(self._clock() * 1000) & 0xFFFFFFFF,
            self.command_target_system,
            self.command_target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            self.POSITION_ONLY_TARGET_TYPE_MASK,
            *target_px4.tolist(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(target_yaw_rad),
            0.0,
        )

    def _receive_statustext(self, message: Any) -> None:
        """Reassembles ordered text chunks, retaining PX4's visible prefix.

        Args:
            message: A decoded MAVLink STATUSTEXT message. Chunk keys include
                sender identity and message ID; gaps and expired tails are
                discarded. PX4 starts IDs at zero, unlike the spec's suggested
                use of zero for standalone texts.
        """
        raw_text = getattr(message, "_text_raw", message.text)
        if isinstance(raw_text, str):
            raw_text = raw_text.encode("utf-8")
        chunk = raw_text.split(b"\x00", 1)[0]
        sequence = int(getattr(message, "chunk_seq", 0))
        key = (
            int(message.get_srcSystem()),
            int(message.get_srcComponent()),
            int(getattr(message, "id", 0)),
        )
        now = self._clock()
        self._statustext_chunks = {
            key_: value for key_, value in self._statustext_chunks.items()
            if now - value[2] <= 2.0
        }
        if sequence == 0:
            combined = chunk
        else:
            previous = self._statustext_chunks.pop(key, None)
            if previous is None or previous[0] + 1 != sequence:
                return
            combined = previous[1] + chunk
        if len(chunk) == 50:
            self._statustext_chunks[key] = (sequence, combined, now)
        else:
            self._statustext_chunks.pop(key, None)
        # PX4 sends exactly-50-byte messages without an extra terminating
        # chunk. Show the cumulative prefix so these remain visible too.
        self.last_statustext = combined.decode("utf-8", errors="replace").strip()
        if self.last_statustext:
            severity = int(getattr(message, "severity", -1))
            print(
                f"PX4 STATUSTEXT[{severity}]: {self.last_statustext}",
                flush=True,
            )

    def poll_status(self) -> None:
        """Polls PX4 heartbeat, command acknowledgement, and status text."""
        if self.command_link is None:
            return
        while True:
            message = self.command_link.recv_match(
                type=["COMMAND_ACK", "STATUSTEXT", "HEARTBEAT"],
                blocking=False,
            )
            if message is None:
                return
            message_type = message.get_type()
            if message_type == "HEARTBEAT":
                base_mode = int(getattr(message, "base_mode", 0))
                custom_mode = int(getattr(message, "custom_mode", 0))
                main_mode = (custom_mode >> 16) & 0xFF
                if main_mode == 0:
                    main_mode = custom_mode
                self.armed = bool(
                    base_mode
                    & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.offboard = main_mode == self.PX4_OFFBOARD_MAIN_MODE
                self.last_px4_heartbeat_wall_time = self._clock()
            elif message_type == "COMMAND_ACK":
                command = int(getattr(message, "command", -1))
                result = int(getattr(message, "result", -1))
                self.last_command_ack = (
                    f"{command}:{result}"
                )
                if command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    self.arm_ack_sequence += 1
                    self.last_arm_ack_result = result
            elif message_type == "STATUSTEXT":
                self._receive_statustext(message)

    def receive_rotor_thrusts(self) -> FloatArray:
        """Returns the most recent four PX4 rotor thrust commands."""
        if self.sim_link is None:
            return self.last_rotor_thrusts.copy()
        while True:
            message = self.sim_link.recv_match(
                type="HIL_ACTUATOR_CONTROLS",
                blocking=False,
            )
            if message is None:
                return self.last_rotor_thrusts.copy()
            controls = np.asarray(message.controls, dtype=np.float64)
            if controls.size < 4 or not np.all(np.isfinite(controls[:4])):
                continue
            self.last_rotor_thrusts = motor_outputs_to_thrusts(
                controls[:4],
                self.rotor_thrust_scales,
            )
            self.last_control_received_wall_time = self._clock()

    def control_is_stale(self, max_age_s: float) -> bool:
        """Reports whether an established actuator stream exceeded its age limit."""
        if max_age_s <= 0.0 or not np.isfinite(max_age_s):
            raise ValueError("max_age_s must be finite and positive")
        if self.last_control_received_wall_time <= 0.0:
            return False
        return (
            self._clock() - self.last_control_received_wall_time
            > max_age_s
        )

    def heartbeat_is_recent(self, max_age_s: float = 2.0) -> bool:
        """Reports whether PX4 recently supplied a command-link heartbeat."""
        if self.last_px4_heartbeat_wall_time <= 0.0:
            return False
        return self._clock() - self.last_px4_heartbeat_wall_time <= max_age_s


@dataclass(frozen=True)
class ConstantWindField:
    """Spatially and temporally constant horizontal world-frame airflow."""

    configured_velocity_world: FloatArray

    def __post_init__(self) -> None:
        """Validates and copies the configured velocity."""
        object.__setattr__(
            self,
            "configured_velocity_world",
            _validated_wind_velocity(self.configured_velocity_world),
        )

    def velocity(self, enabled: bool) -> FloatArray:
        """Returns the active wind velocity in world coordinates.

        Args:
            enabled: Whether the configured wind field is active.

        Returns:
            Configured wind velocity when enabled, otherwise a zero vector.
        """
        if not enabled:
            return np.zeros(3, dtype=np.float64)
        return self.configured_velocity_world.copy()


@dataclass(frozen=True)
class TrajectorySample:
    """One sample of a predefined world-frame position trajectory."""

    position_world: FloatArray
    phase: float


@dataclass(frozen=True)
class PredefinedXYTrajectory:
    """Deterministic bounded trajectory in the world XY plane."""

    name: str
    center_xy: FloatArray
    altitude: float
    start_time: float
    duration: float
    size_m: float

    def __post_init__(self) -> None:
        """Validates scalar parameters and the trajectory name."""
        center = np.asarray(self.center_xy, dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center_xy must be a finite vector with shape (2,)")
        scalar_values = np.array(
            [self.altitude, self.start_time, self.duration, self.size_m],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalar_values)):
            raise ValueError("trajectory scalar values must be finite")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.size_m <= 0.0:
            raise ValueError("size_m must be positive")
        object.__setattr__(self, "center_xy", center.copy())
        self._offset_at_phase(0.0)

    @classmethod
    def from_start(
        cls,
        name: str,
        start_xy: Sequence[float],
        altitude: float,
        start_time: float,
        duration: float,
        size_m: float,
    ) -> "PredefinedXYTrajectory":
        """Creates a trajectory whose first sample matches ``start_xy``.

        Args:
            name: One of ``figure8``, ``ellipse``, ``circle``, or ``square``.
            start_xy: Required world-frame starting position.
            altitude: Fixed world-frame target altitude in meters.
            start_time: Simulation start time in seconds.
            duration: Playback duration in seconds.
            size_m: Overall horizontal path size in meters.

        Returns:
            A trajectory translated to start continuously at ``start_xy``.
        """
        start = np.asarray(start_xy, dtype=np.float64)
        if start.shape != (2,) or not np.all(np.isfinite(start)):
            raise ValueError("start_xy must be a finite vector with shape (2,)")
        zero_center = cls(
            name=name,
            center_xy=np.zeros(2, dtype=np.float64),
            altitude=altitude,
            start_time=start_time,
            duration=duration,
            size_m=size_m,
        )
        initial_offset = zero_center._offset_at_phase(0.0)
        return cls(
            name=name,
            center_xy=start - initial_offset,
            altitude=altitude,
            start_time=start_time,
            duration=duration,
            size_m=size_m,
        )

    def sample(self, sim_time: float) -> TrajectorySample | None:
        """Samples the path at a simulation time.

        Args:
            sim_time: Current simulation time in seconds.

        Returns:
            A position sample, or ``None`` after the playback duration.
        """
        if not np.isfinite(sim_time):
            raise ValueError("sim_time must be finite")
        elapsed = float(sim_time) - self.start_time
        if elapsed >= self.duration:
            return None
        phase = float(np.clip(elapsed / self.duration, 0.0, 1.0))
        offset = self._offset_at_phase(phase)
        position = np.array(
            [
                self.center_xy[0] + offset[0],
                self.center_xy[1] + offset[1],
                self.altitude,
            ],
            dtype=np.float64,
        )
        return TrajectorySample(position_world=position, phase=phase)

    def _offset_at_phase(self, phase: float) -> FloatArray:
        """Returns the horizontal path offset at normalized ``phase``."""
        normalized_name = self.name.lower().replace("-", "").replace("_", "")
        angle = 2.0 * np.pi * float(phase)
        half_size = 0.5 * self.size_m
        if normalized_name in ("figure8", "fig8"):
            return np.array(
                [
                    half_size * np.sin(angle),
                    0.5 * half_size * np.sin(2.0 * angle),
                ],
                dtype=np.float64,
            )
        if normalized_name == "ellipse":
            return np.array(
                [
                    half_size * np.cos(angle),
                    0.3 * self.size_m * np.sin(angle),
                ],
                dtype=np.float64,
            )
        if normalized_name == "circle":
            return np.array(
                [half_size * np.cos(angle), half_size * np.sin(angle)],
                dtype=np.float64,
            )
        if normalized_name == "square":
            return self._square_offset_at_phase(phase, half_size)
        raise ValueError(f"unsupported predefined trajectory: {self.name}")

    @staticmethod
    def _square_offset_at_phase(phase: float, half_size: float) -> FloatArray:
        """Returns a constant-speed square-perimeter offset."""
        corners = np.array(
            [
                [half_size, half_size],
                [-half_size, half_size],
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
            ],
            dtype=np.float64,
        )
        segment_position = 4.0 * float(np.clip(phase, 0.0, 1.0))
        segment_index = min(int(np.floor(segment_position)), 3)
        segment_fraction = segment_position - float(segment_index)
        return corners[segment_index] + segment_fraction * (
            corners[segment_index + 1] - corners[segment_index]
        )


@dataclass
class RealtimePredefinedTrajectoryPlot:
    """Realtime XY target and actual trajectory plot state."""

    figure: Figure
    axis: Axes
    target_line: Line2D
    actual_line: Line2D
    actual_xy: list[FloatArray] = field(default_factory=list)
    last_update_sim_time: float = -np.inf

    def close(self) -> None:
        """Closes the realtime trajectory figure."""
        plt.close(self.figure)


def sample_predefined_trajectory_target_xy(
    trajectory: PredefinedXYTrajectory,
    sample_count: int = TRAJECTORY_PLOT_TARGET_SAMPLE_COUNT,
) -> FloatArray:
    """Samples the complete predefined target path for plotting.

    Args:
        trajectory: Predefined world-frame XY trajectory.
        sample_count: Number of uniformly spaced path samples.

    Returns:
        Target XY positions with shape ``(sample_count, 2)``.

    Raises:
        ValueError: If fewer than two samples are requested.
    """
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    phases = np.linspace(0.0, 1.0, sample_count, endpoint=True)
    return np.asarray(
        [
            trajectory.center_xy + trajectory._offset_at_phase(float(phase))
            for phase in phases
        ],
        dtype=np.float64,
    )


def create_realtime_predefined_trajectory_plot(
    trajectory: PredefinedXYTrajectory,
) -> RealtimePredefinedTrajectoryPlot:
    """Creates a non-blocking XY plot for target and actual trajectories.

    Args:
        trajectory: Active predefined trajectory.

    Returns:
        Plot artists and actual-position history used for realtime updates.
    """
    target_xy = sample_predefined_trajectory_target_xy(trajectory)
    figure, axis = plt.subplots(figsize=(6.4, 6.0))
    figure.canvas.manager.set_window_title(
        "UAV Predefined Trajectory (Realtime)"
    )
    (target_line,) = axis.plot(
        target_xy[:, 0],
        target_xy[:, 1],
        linestyle="--",
        linewidth=1.7,
        color="0.35",
        label="target",
    )
    (actual_line,) = axis.plot(
        [],
        [],
        linestyle="-",
        linewidth=1.5,
        color="#b91c1c",
        label="actual",
    )
    axis.set_title(f"{trajectory.name.title()} XY Trajectory")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.grid(True, alpha=0.35)
    axis.legend(loc="best")
    axis.set_aspect("equal", adjustable="box")
    x_span = float(np.ptp(target_xy[:, 0]))
    y_span = float(np.ptp(target_xy[:, 1]))
    padding = max(0.25, 0.15 * max(x_span, y_span))
    axis.set_xlim(
        float(np.min(target_xy[:, 0]) - padding),
        float(np.max(target_xy[:, 0]) + padding),
    )
    axis.set_ylim(
        float(np.min(target_xy[:, 1]) - padding),
        float(np.max(target_xy[:, 1]) + padding),
    )
    figure.tight_layout()
    return RealtimePredefinedTrajectoryPlot(
        figure=figure,
        axis=axis,
        target_line=target_line,
        actual_line=actual_line,
    )


def show_realtime_figure(figure: Figure) -> None:
    """Shows a Matplotlib figure without blocking the simulation loop."""
    figure.show()
    figure.canvas.flush_events()


def update_realtime_predefined_trajectory_plot(
    plot: RealtimePredefinedTrajectoryPlot,
    position_world: Sequence[float],
    sim_time: float,
) -> None:
    """Appends actual XY position and periodically redraws the plot.

    Args:
        plot: Realtime plot state to update.
        position_world: Current world-frame UAV XYZ position.
        sim_time: Current MuJoCo simulation time in seconds.

    Raises:
        ValueError: If ``sim_time`` is non-finite.
    """
    if not np.isfinite(sim_time):
        raise ValueError("sim_time must be finite")
    position = _validated_vector3(position_world, "position_world")
    if not plt.fignum_exists(plot.figure.number):
        return
    plot.actual_xy.append(position[:2].copy())
    if (
        float(sim_time) - plot.last_update_sim_time
        < TRAJECTORY_PLOT_UPDATE_PERIOD_S
    ):
        return
    actual_array = np.asarray(plot.actual_xy, dtype=np.float64)
    plot.actual_line.set_data(actual_array[:, 0], actual_array[:, 1])
    plot.last_update_sim_time = float(sim_time)
    plot.figure.canvas.draw_idle()
    plot.figure.canvas.flush_events()


class FlightState(enum.Enum):
    """High-level flight states displayed by the control panel."""

    DISARMED = enum.auto()
    ARMED = enum.auto()
    TAKEOFF = enum.auto()
    FLYING = enum.auto()
    LANDING = enum.auto()
    EMERGENCY_STOP = enum.auto()


class UiCommandType(enum.Enum):
    """Commands accepted from the graphical control panel."""

    UNLOCK = enum.auto()
    TAKEOFF = enum.auto()
    LAND = enum.auto()
    EMERGENCY_STOP = enum.auto()
    TOGGLE_WIND = enum.auto()
    SET_WIND = enum.auto()
    START_TRAJECTORY = enum.auto()
    MOVE_POSITION = enum.auto()
    EXIT = enum.auto()


@dataclass(frozen=True)
class UiCommand:
    """One queued user-interface command."""

    command_type: UiCommandType
    payload: str | tuple[float, float, float] | None = None


class ControlAction(enum.Enum):
    """Side effects requested by the pure flight state machine."""

    REQUEST_ARM = enum.auto()
    REQUEST_DISARM = enum.auto()
    ENTER_OFFBOARD = enum.auto()
    SEND_TAKEOFF_TARGET = enum.auto()
    SEND_LANDING_TARGET = enum.auto()
    SEND_POSITION_TARGET = enum.auto()
    START_TRAJECTORY = enum.auto()
    ZERO_THRUST = enum.auto()


@dataclass
class FlightControlState:
    """Explicit flight-mode state independent of MuJoCo and MAVLink I/O."""

    initial_position_world: FloatArray
    takeoff_height_m: float
    flight_state: FlightState = FlightState.DISARMED
    armed: bool = False
    offboard: bool = False
    wind_enabled: bool = False
    wind_configured: bool = False
    active_trajectory_name: str = "none"

    def __post_init__(self) -> None:
        """Validates initial state and initializes the position target."""
        initial_position = _validated_vector3(
            self.initial_position_world,
            "initial_position_world",
        )
        if not np.isfinite(self.takeoff_height_m) or self.takeoff_height_m <= 0.0:
            raise ValueError("takeoff_height_m must be finite and positive")
        self.initial_position_world = initial_position
        self.current_position_world = initial_position.copy()
        self.target_position_world = initial_position.copy()

    def update_px4_status(self, armed: bool, offboard: bool) -> None:
        """Updates confirmed PX4 state without issuing commands.

        Args:
            armed: Confirmed PX4 Arm flag.
            offboard: Confirmed PX4 Offboard main-mode flag.
        """
        self.armed = bool(armed)
        self.offboard = bool(offboard)
        if self.flight_state is FlightState.EMERGENCY_STOP:
            return
        if not self.armed:
            self.flight_state = FlightState.DISARMED
        elif self.flight_state is FlightState.DISARMED:
            self.flight_state = FlightState.ARMED

    def update_position(self, position_world: Sequence[float]) -> None:
        """Updates position and completes takeoff or landing transitions."""
        self.current_position_world = _validated_vector3(
            position_world,
            "position_world",
        )
        if (
            self.flight_state is FlightState.TAKEOFF
            and self.current_position_world[2] >= self.target_position_world[2] - 0.2
        ):
            self.flight_state = FlightState.FLYING
        elif (
            self.flight_state is FlightState.LANDING
            and self.current_position_world[2]
            <= self.initial_position_world[2] + 0.05
        ):
            self.flight_state = (
                FlightState.ARMED if self.armed else FlightState.DISARMED
            )

    def handle(self, command: UiCommand) -> list[ControlAction]:
        """Applies a UI command and returns requested external actions.

        Args:
            command: Typed UI command to process.

        Returns:
            Ordered side effects for the application loop to execute.
        """
        if self.flight_state is FlightState.EMERGENCY_STOP:
            return []

        if command.command_type is UiCommandType.UNLOCK:
            if self.flight_state is not FlightState.DISARMED:
                return []
            return [ControlAction.REQUEST_ARM]

        if command.command_type is UiCommandType.TAKEOFF:
            if self.flight_state is not FlightState.ARMED:
                return []
            self.target_position_world = self.current_position_world.copy()
            self.target_position_world[2] = (
                self.initial_position_world[2] + self.takeoff_height_m
            )
            self.flight_state = FlightState.TAKEOFF
            return [
                ControlAction.ENTER_OFFBOARD,
                ControlAction.SEND_TAKEOFF_TARGET,
            ]

        if command.command_type is UiCommandType.MOVE_POSITION:
            if (
                self.flight_state is not FlightState.FLYING
                or self.active_trajectory_name != "none"
            ):
                return []
            delta_world = _validated_vector3(
                command.payload,
                "move_position_delta",
            )
            self.target_position_world = (
                self.target_position_world + delta_world
            )
            return [ControlAction.SEND_POSITION_TARGET]

        if command.command_type is UiCommandType.START_TRAJECTORY:
            if self.flight_state is not FlightState.FLYING:
                return []
            trajectory_name = str(command.payload or "").lower()
            if trajectory_name not in ("figure8", "ellipse", "circle", "square"):
                return []
            self.active_trajectory_name = trajectory_name
            return [ControlAction.START_TRAJECTORY]

        if command.command_type is UiCommandType.LAND:
            if self.flight_state not in (FlightState.TAKEOFF, FlightState.FLYING):
                return []
            self.active_trajectory_name = "none"
            self.target_position_world = self.current_position_world.copy()
            self.target_position_world[2] = self.initial_position_world[2]
            self.flight_state = FlightState.LANDING
            return [ControlAction.SEND_LANDING_TARGET]

        if command.command_type is UiCommandType.TOGGLE_WIND:
            if self.wind_configured:
                self.wind_enabled = not self.wind_enabled
            return []

        if command.command_type is UiCommandType.EMERGENCY_STOP:
            self.active_trajectory_name = "none"
            self.flight_state = FlightState.EMERGENCY_STOP
            return [
                ControlAction.ZERO_THRUST,
                ControlAction.REQUEST_DISARM,
            ]

        return []


@dataclass(frozen=True)
class PanelSnapshot:
    """Immutable simulation state consumed by ``ControlPanel``."""

    px4_connected: bool
    armed: bool
    offboard: bool
    flight_state: str
    trajectory: str
    position_world: FloatArray
    target_world: FloatArray
    configured_wind_world: FloatArray | None
    active_wind_world: FloatArray
    filtered_residual_acceleration: FloatArray
    pinn_force_world: FloatArray
    wind_enabled: bool
    last_error: str = "none"
    unlock_pending: bool = False
    wind_backend: str = "pinn-x4"
    simulation_time_s: float = 0.0


@dataclass
class TriangleMovementControl:
    """One triangular world-frame position-jog control."""

    axis: Axes
    patch: Polygon
    delta_world: tuple[float, float, float]
    pointer: str
    base_color: str
    hover_color: str
    hovered: bool = False


class ControlPanel:
    """Compact Matplotlib command panel."""

    ACTIVE_BUTTON_COLOR = "#b7d7f0"
    ACTIVE_BUTTON_HOVER_COLOR = "#9cc8ea"
    INACTIVE_BUTTON_COLOR = "#e2e2e2"
    INACTIVE_BUTTON_HOVER_COLOR = "#cdcdcd"
    EMERGENCY_BUTTON_COLOR = "#ffb3b3"
    EMERGENCY_BUTTON_HOVER_COLOR = "#ff9999"
    PENDING_BUTTON_COLOR = "#fde68a"
    PENDING_BUTTON_HOVER_COLOR = "#fcd34d"
    STATE_DRIVEN_BUTTON_NAMES = (
        "unlock",
        "takeoff",
        "land",
        "wind",
        "figure8",
        "ellipse",
        "circle",
        "square",
    )

    def __init__(
        self,
        command_queue: queue.SimpleQueue[UiCommand],
        initial_wind_world: Sequence[float] | None = None,
    ) -> None:
        """Build controls whose callbacks enqueue typed commands.

        Args:
            command_queue: Commands consumed by the simulation loop.
            initial_wind_world: Optional [X, Y, 0] draft; Apply is required.
        """
        self.command_queue = command_queue
        self.figure = plt.figure(figsize=(7.4, 5.0), facecolor="#f5f5f5")
        self.figure.canvas.manager.set_window_title("PX4 Wind Simulation Control")
        self.figure.suptitle(
            "PX4 MuJoCo Wind Simulation",
            fontsize=13,
            fontweight="bold",
        )
        self.buttons: dict[str, Button] = {}
        button_specs = [
            ("unlock", "Unlock", 0.05, 0.78, self._on_unlock),
            ("takeoff", "Takeoff", 0.20, 0.78, self._on_takeoff),
            ("land", "Land", 0.35, 0.78, self._on_land),
            ("emergency", "Emergency", 0.50, 0.78, self._on_emergency),
            ("wind", "Wind OFF", 0.72, 0.78, self._on_toggle_wind),
        ]
        for name, label, left, bottom, callback in button_specs:
            axis = self.figure.add_axes([left, bottom, 0.13, 0.09])
            button = Button(axis, label)
            button.on_clicked(callback)
            self.buttons[name] = button
        trajectory_specs = [
            ("figure8", "Figure-8", 0.05),
            ("ellipse", "Ellipse", 0.22),
            ("circle", "Circle", 0.39),
            ("square", "Square", 0.56),
        ]
        for name, label, left in trajectory_specs:
            axis = self.figure.add_axes([left, 0.62, 0.14, 0.09])
            button = Button(
                axis,
                label,
            )
            button.on_clicked(
                lambda _event, trajectory_name=name: self._on_trajectory(
                    trajectory_name
                )
            )
            self.buttons[name] = button
        self.figure.text(
            0.08,
            0.555,
            "Position Jog (World, 0.5 m)",
            fontsize=10,
            fontweight="bold",
            color="#2a2a2a",
        )
        self.triangle_controls: dict[str, TriangleMovementControl] = {}
        jog_specs = [
            (
                "x_plus",
                "X+",
                [0.245, 0.42, 0.10, 0.11],
                (0.5, 0.0, 0.0),
                "up",
                "#dbeafe",
                "#bfdbfe",
            ),
            (
                "x_minus",
                "X-",
                [0.245, 0.16, 0.10, 0.11],
                (-0.5, 0.0, 0.0),
                "down",
                "#dbeafe",
                "#bfdbfe",
            ),
            (
                "y_plus",
                "Y+",
                [0.13, 0.29, 0.10, 0.11],
                (0.0, 0.5, 0.0),
                "left",
                "#dcfce7",
                "#bbf7d0",
            ),
            (
                "y_minus",
                "Y-",
                [0.36, 0.29, 0.10, 0.11],
                (0.0, -0.5, 0.0),
                "right",
                "#dcfce7",
                "#bbf7d0",
            ),
            (
                "z_plus",
                "Z+",
                [0.13, 0.02, 0.10, 0.11],
                (0.0, 0.0, 0.5),
                "up",
                "#ffedd5",
                "#fed7aa",
            ),
            (
                "z_minus",
                "Z-",
                [0.36, 0.02, 0.10, 0.11],
                (0.0, 0.0, -0.5),
                "down",
                "#ffedd5",
                "#fed7aa",
            ),
        ]
        for (
            name,
            label,
            rect,
            delta_world,
            pointer,
            base_color,
            hover_color,
        ) in jog_specs:
            self._add_triangle_control(
                name=name,
                label=label,
                rect=rect,
                delta_world=delta_world,
                pointer=pointer,
                base_color=base_color,
                hover_color=hover_color,
            )
        self._motion_callback_id = self.figure.canvas.mpl_connect(
            "motion_notify_event",
            self._on_panel_motion,
        )
        self._click_callback_id = self.figure.canvas.mpl_connect(
            "button_press_event",
            self._on_panel_click,
        )
        exit_axis = self.figure.add_axes([0.76, 0.085, 0.18, 0.06])
        exit_button = Button(exit_axis, "Exit")
        exit_button.on_clicked(self._on_exit)
        self.buttons["exit"] = exit_button
        self._set_button_active("exit", active=False)
        for button_name in self.STATE_DRIVEN_BUTTON_NAMES:
            self._set_button_active(button_name, active=False)
        self._set_emergency_palette()

        status_axis = self.figure.add_axes([0.52, 0.26, 0.44, 0.29])
        status_axis.set_axis_off()
        self.status_text = status_axis.text(
            0.0, 1.0, "Waiting for PX4 heartbeat", va="top", fontsize=9,
        )
        self._create_wind_controls(initial_wind_world)

    def _create_wind_controls(
        self, initial_wind_world: Sequence[float] | None,
    ) -> None:
        """Build the signed world-velocity editor below the diagnostics.

        Args:
            initial_wind_world: Optional draft, not accepted wind settings.
        """
        draft = (
            _validated_wind_velocity(initial_wind_world)
            if initial_wind_world is not None else None
        )
        self._displayed_wind_world: FloatArray | None = None
        self._wind_inputs_dirty = False
        self._wind_input_error = ""
        self._pending_wind_world: FloatArray | None = None
        self._wind_enabled = False
        self.buttons["wind"].set_active(False)
        self.buttons["wind"].label.set_alpha(0.4)
        self.figure.text(
            0.52, 0.235, "Horizontal wind (world, <=3 m/s)", fontsize=9,
        )
        self.wind_feedback_text = self.figure.text(
            0.52, 0.023, "", fontsize=8, va="bottom", color="#1f2937",
        )
        self.wind_inputs: list[TextBox] = []
        for index, (label, left) in enumerate(zip(
            ("X", "Y"), (0.56, 0.79),
        )):
            axis = self.figure.add_axes([left, 0.17, 0.16, 0.05])
            initial_text = "" if draft is None else str(float(draft[index]))
            text_box = TextBox(axis, label, initial=initial_text)
            text_box.on_text_change(self._on_wind_input_changed)
            self.wind_inputs.append(text_box)
        apply_axis = self.figure.add_axes([0.52, 0.085, 0.20, 0.06])
        apply_button = Button(apply_axis, "Apply Wind")
        apply_button.on_clicked(self._on_apply_wind)
        self.buttons["apply_wind"] = apply_button
        self._render_wind_feedback()

    def _on_wind_input_changed(self, _text: str) -> None:
        """Mark edits as drafts so status refreshes cannot overwrite them."""
        self._wind_inputs_dirty = True
        self._wind_input_error = ""
        self._render_wind_feedback()

    def _on_apply_wind(self, _event: object) -> None:
        """Validate a complete velocity vector and queue it atomically."""
        try:
            velocity = _validated_wind_velocity(
                [box.text.strip() for box in self.wind_inputs] + [0.0],
            )
        except ValueError as error:
            self._wind_input_error = str(error)
            self._render_wind_feedback()
            return
        self.command_queue.put(UiCommand(
            UiCommandType.SET_WIND,
            (float(velocity[0]), float(velocity[1]), float(velocity[2])),
        ))
        self._pending_wind_world = velocity
        self._wind_inputs_dirty = False
        self._wind_input_error = ""
        self._render_wind_feedback()

    def _refresh_wind_settings(self, snapshot: PanelSnapshot) -> None:
        """Show accepted settings while preserving any newer draft.

        Args:
            snapshot: Authoritative state from the simulation loop.
        """
        configured = snapshot.configured_wind_world
        if not np.array_equal(configured, self._displayed_wind_world):
            self._displayed_wind_world = (
                configured.copy() if configured is not None else None
            )
            if not self._wind_inputs_dirty and configured is not None:
                for box, value in zip(self.wind_inputs, configured):
                    box.set_val(str(float(value)))
                self._wind_inputs_dirty = False
        if self._pending_wind_world is not None and np.array_equal(
            configured, self._pending_wind_world,
        ):
            self._pending_wind_world = None
        self._wind_enabled = snapshot.wind_enabled
        ready = configured is not None
        self.buttons["wind"].set_active(ready)
        self.buttons["wind"].label.set_alpha(1.0 if ready else 0.4)
        self._render_wind_feedback()

    def _render_wind_feedback(self) -> None:
        """Display the accepted vector, magnitude and editor status."""
        configured = self._displayed_wind_world
        header = "Wind not configured."
        if configured is not None:
            components = ", ".join(f"{value:.3g}" for value in configured[:2])
            header = f"Set: [{components}] m/s"
        if self._wind_input_error:
            status = self._wind_input_error
        elif self._wind_inputs_dirty:
            status = "Editing; click Apply Wind"
        elif self._pending_wind_world is not None:
            status = "Update queued"
        elif configured is None:
            status = "Enter X/Y, then Apply Wind"
        else:
            speed = float(np.linalg.norm(configured))
            enabled = "ON" if self._wind_enabled else "OFF"
            status = f"Speed: {speed:.3g} m/s | Wind {enabled}"
        self.wind_feedback_text.set_text(f"{header}\n{status}")
        self.wind_feedback_text.set_color(
            "#b91c1c" if self._wind_input_error else "#1f2937"
        )

    @staticmethod
    def _triangle_vertices(pointer: str) -> list[tuple[float, float]]:
        """Returns normalized triangle vertices for one pointer direction.

        Args:
            pointer: One of ``up``, ``down``, ``left``, or ``right``.

        Returns:
            Three vertices in axes-normalized coordinates.

        Raises:
            ValueError: If ``pointer`` is unsupported.
        """
        vertices = {
            "up": [(0.5, 0.9), (0.12, 0.12), (0.88, 0.12)],
            "down": [(0.12, 0.88), (0.88, 0.88), (0.5, 0.1)],
            "left": [(0.1, 0.5), (0.88, 0.88), (0.88, 0.12)],
            "right": [(0.12, 0.88), (0.9, 0.5), (0.12, 0.12)],
        }
        try:
            return vertices[pointer]
        except KeyError as error:
            raise ValueError(f"unsupported triangle pointer: {pointer}") from error

    def _add_triangle_control(
        self,
        name: str,
        label: str,
        rect: list[float],
        delta_world: tuple[float, float, float],
        pointer: str,
        base_color: str,
        hover_color: str,
    ) -> None:
        """Adds one clickable triangular movement control to the panel.

        Args:
            name: Stable key used to access the control.
            label: Text drawn at the triangle center.
            rect: Matplotlib axes rectangle in figure coordinates.
            delta_world: XYZ target increment in metres.
            pointer: Direction in which the triangle points.
            base_color: Normal face color.
            hover_color: Face color while the pointer is over the triangle.
        """
        axis = self.figure.add_axes(rect)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#f5f5f5")
        for spine in axis.spines.values():
            spine.set_visible(False)
        patch = Polygon(
            self._triangle_vertices(pointer),
            closed=True,
            facecolor=base_color,
            edgecolor="#606060",
            linewidth=1.2,
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10.8,
            fontweight="bold",
            color="#1f1f1f",
        )
        self.triangle_controls[name] = TriangleMovementControl(
            axis=axis,
            patch=patch,
            delta_world=delta_world,
            pointer=pointer,
            base_color=base_color,
            hover_color=hover_color,
        )

    def _on_panel_motion(self, event: MouseEvent) -> None:
        """Updates triangle colors when the mouse enters or leaves them."""
        changed = False
        for control in self.triangle_controls.values():
            is_hovered = bool(
                event.inaxes is control.axis
                and control.patch.contains(event)[0]
            )
            if is_hovered != control.hovered:
                control.hovered = is_hovered
                color = (
                    control.hover_color
                    if is_hovered
                    else control.base_color
                )
                control.patch.set_facecolor(color)
                changed = True
        if changed:
            self.figure.canvas.draw_idle()

    def _on_panel_click(self, event: MouseEvent) -> None:
        """Enqueues a movement command for a clicked triangle."""
        if event.button != 1:
            return
        for control in self.triangle_controls.values():
            if (
                event.inaxes is control.axis
                and control.patch.contains(event)[0]
            ):
                self._on_move_position(control.delta_world)
                break

    @property
    def is_open(self) -> bool:
        """Returns whether the panel figure still exists."""
        return bool(plt.fignum_exists(self.figure.number))

    def _set_button_active(self, name: str, active: bool) -> None:
        """Sets one state-driven button's active or inactive palette.

        Args:
            name: Key of the Matplotlib button to update.
            active: Whether to apply the light-blue active palette.
        """
        button = self.buttons[name]
        button.color = (
            self.ACTIVE_BUTTON_COLOR
            if active
            else self.INACTIVE_BUTTON_COLOR
        )
        button.hovercolor = (
            self.ACTIVE_BUTTON_HOVER_COLOR
            if active
            else self.INACTIVE_BUTTON_HOVER_COLOR
        )
        button.ax.set_facecolor(button.color)

    def _set_emergency_palette(self) -> None:
        """Restores the emergency button's permanent red warning palette."""
        button = self.buttons["emergency"]
        button.color = self.EMERGENCY_BUTTON_COLOR
        button.hovercolor = self.EMERGENCY_BUTTON_HOVER_COLOR
        button.ax.set_facecolor(button.color)

    def _set_unlock_palette(self, armed: bool, pending: bool) -> None:
        """Shows confirmed armed, pending request, or inactive unlock state."""
        if armed:
            self._set_button_active("unlock", active=True)
            return
        if not pending:
            self._set_button_active("unlock", active=False)
            return
        button = self.buttons["unlock"]
        button.color = self.PENDING_BUTTON_COLOR
        button.hovercolor = self.PENDING_BUTTON_HOVER_COLOR
        button.ax.set_facecolor(button.color)

    def _on_unlock(self, _event: object) -> None:
        self.command_queue.put(UiCommand(UiCommandType.UNLOCK))

    def _on_takeoff(self, _event: object) -> None:
        self.command_queue.put(UiCommand(UiCommandType.TAKEOFF))

    def _on_land(self, _event: object) -> None:
        self.command_queue.put(UiCommand(UiCommandType.LAND))

    def _on_emergency(self, _event: object) -> None:
        self.command_queue.put(UiCommand(UiCommandType.EMERGENCY_STOP))

    def _on_toggle_wind(self, _event: object) -> None:
        if self._displayed_wind_world is not None:
            self.command_queue.put(UiCommand(UiCommandType.TOGGLE_WIND))

    def _on_move_position(
        self,
        delta_world: tuple[float, float, float],
    ) -> None:
        """Enqueues one fixed world-frame position-target increment.

        Args:
            delta_world: Immutable XYZ target increment in metres.
        """
        self.command_queue.put(
            UiCommand(UiCommandType.MOVE_POSITION, delta_world)
        )

    def _on_exit(self, _event: object) -> None:
        """Enqueues one request to shut down PX4 and exit the simulation."""
        self.command_queue.put(UiCommand(UiCommandType.EXIT))

    def _on_trajectory(self, name: str) -> None:
        self.command_queue.put(
            UiCommand(UiCommandType.START_TRAJECTORY, name)
        )

    def update(self, snapshot: PanelSnapshot) -> None:
        """Refreshes button labels and confirmed-state highlights."""
        link_status = "connected" if snapshot.px4_connected else "no heartbeat"
        unlock_status = (
            "waiting for confirmation" if snapshot.unlock_pending
            else "armed" if snapshot.armed else "disarmed"
        )
        lines = [f"PX4: {link_status}", f"Unlock: {unlock_status}",
                 f"Wind backend: {snapshot.wind_backend}",
                 f"Sim time: {snapshot.simulation_time_s:.1f} s"]
        if snapshot.last_error != "none":
            lines.append(textwrap.fill(snapshot.last_error, width=38))
        self.status_text.set_text("\n".join(lines))
        self.status_text.set_color(
            "#b91c1c" if snapshot.last_error != "none" else "#1f2937"
        )
        wind_button = self.buttons["wind"]
        wind_button.label.set_text(
            "Wind ON" if snapshot.wind_enabled else "Wind OFF"
        )
        self._refresh_wind_settings(snapshot)
        self._set_unlock_palette(
            armed=snapshot.armed,
            pending=snapshot.unlock_pending,
        )
        self._set_button_active(
            "takeoff",
            snapshot.flight_state == "TAKEOFF",
        )
        self._set_button_active(
            "land",
            snapshot.flight_state == "LANDING",
        )
        self._set_button_active("wind", snapshot.wind_enabled)
        for trajectory_name in ("figure8", "ellipse", "circle", "square"):
            self._set_button_active(
                trajectory_name,
                snapshot.trajectory == trajectory_name,
            )
        self._set_emergency_palette()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self) -> None:
        """Closes the panel figure."""
        plt.close(self.figure)


@dataclass(frozen=True)
class SimulationConfig:
    """Validated command-line configuration for one experiment run."""

    xml_path: Path
    params_path: Path
    sim_tcp_port: int
    command_udp_port: int
    takeoff_height_m: float
    wind_velocity_world: FloatArray | None
    trajectory_size_m: float
    trajectory_duration_s: float
    figure8_duration_s: float
    viewer_enabled: bool
    wind_backend: str = "pinn-x4"
    reference_config: Path | None = None
    run_log: Path | None = None
    pinn_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        """Validates numeric and path configuration."""
        if self.wind_backend not in ("pinn", "analytic", "pinn-x4"):
            raise ValueError("wind_backend must be pinn, analytic or pinn-x4")
        if self.wind_backend in ("analytic", "pinn-x4") and self.reference_config is None:
            object.__setattr__(self, "reference_config", default_wind_reference_config())
        if self.wind_backend == "pinn" and self.reference_config is not None:
            raise ValueError("--reference-config requires analytic or pinn-x4 backend")
        if self.wind_backend == "pinn-x4" and self.pinn_checkpoint is None:
            object.__setattr__(self, "pinn_checkpoint", default_x4_pinn_checkpoint())
        if self.wind_backend != "pinn-x4" and self.pinn_checkpoint is not None:
            raise ValueError("--pinn-checkpoint requires pinn-x4 backend")
        if self.wind_velocity_world is not None:
            object.__setattr__(
                self, "wind_velocity_world",
                _validated_wind_velocity(self.wind_velocity_world),
            )
        positive_values = np.array(
            [
                self.takeoff_height_m,
                self.trajectory_size_m,
                self.trajectory_duration_s,
                self.figure8_duration_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positive_values)) or np.any(
            positive_values <= 0.0
        ):
            raise ValueError("height, trajectory size, and durations must be positive")
        for port_name, port in (
            ("sim_tcp_port", self.sim_tcp_port),
            ("command_udp_port", self.command_udp_port),
        ):
            if not 1 <= int(port) <= 65_535:
                raise ValueError(f"{port_name} must be in [1, 65535]")


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the standalone wind-simulation command-line parser."""
    parser = argparse.ArgumentParser(
        description="PX4 SITL closed-loop MuJoCo constant-wind experiment",
    )
    parser.add_argument("--xml-path", type=Path, default=default_xml_path())
    parser.add_argument("--params-path", type=Path, default=default_params_path())
    parser.add_argument("--sim-tcp-port", type=int, default=4560)
    parser.add_argument("--command-udp-port", type=int, default=14580)
    parser.add_argument("--takeoff-height", type=float, default=1.0)
    parser.add_argument(
        "--wind-backend", choices=("pinn", "analytic", "pinn-x4"), default="pinn-x4",
        help="Default: bundled pinn-x4. Explicit fallbacks: analytic or legacy pinn.",
    )
    parser.add_argument(
        "--reference-config", type=Path,
        help="Override the bundled frozen reference for analytic or pinn-x4.",
    )
    parser.add_argument(
        "--pinn-checkpoint", type=Path,
        help="Override the bundled X4 v1 candidate.pt (pinn-x4 backend only).",
    )
    parser.add_argument(
        "--run-log", type=Path,
        help="Optional new JSONL path for 50 Hz comparison observations.",
    )
    for axis in ("x", "y"):
        parser.add_argument(
            f"--wind-{axis}", type=float, default=None,
            help="Prefill a world-frame wind draft in m/s; Apply Wind is required.",
        )
    parser.add_argument("--trajectory-size", type=float, default=1.0)
    parser.add_argument("--trajectory-duration", type=float, default=20.0)
    parser.add_argument("--figure8-duration", type=float, default=30.0)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable the MuJoCo passive viewer while retaining the control panel.",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> SimulationConfig:
    """Parses command-line arguments into a validated configuration."""
    args = build_argument_parser().parse_args(argv)
    components = [args.wind_x, args.wind_y]
    wind_draft = (
        None if all(value is None for value in components)
        else np.array(
            [0.0 if value is None else value for value in components] + [0.0],
            dtype=np.float64,
        )
    )
    return SimulationConfig(
        xml_path=args.xml_path.resolve(),
        params_path=args.params_path.resolve(),
        sim_tcp_port=args.sim_tcp_port,
        command_udp_port=args.command_udp_port,
        takeoff_height_m=args.takeoff_height,
        wind_velocity_world=wind_draft,
        trajectory_size_m=args.trajectory_size,
        trajectory_duration_s=args.trajectory_duration,
        figure8_duration_s=args.figure8_duration,
        viewer_enabled=not args.no_viewer,
        wind_backend=args.wind_backend,
        reference_config=(
            args.reference_config.resolve() if args.reference_config else None
        ),
        run_log=args.run_log.resolve() if args.run_log else None,
        pinn_checkpoint=(args.pinn_checkpoint.resolve() if args.pinn_checkpoint else None),
    )


def apply_external_wind_force(
    data: Any,
    body_id: int,
    force_world: Sequence[float],
) -> None:
    """Clears stale external wrench and applies a world-frame wind force.

    Args:
        data: Writable MuJoCo data object.
        body_id: Body receiving the external wrench.
        force_world: Three-axis world-frame force in newtons.
    """
    force = _validated_vector3(force_world, "force_world")
    data.xfrc_applied[int(body_id), 0:6] = 0.0
    data.xfrc_applied[int(body_id), 0:3] = force


def enforce_stale_control_safety(
    data: Any,
    bindings: Any,
    flight_control: FlightControlState,
    bridge: Any,
    max_age_s: float = 0.5,
) -> bool:
    """Latches emergency stop if the in-flight PX4 actuator stream is stale."""
    airborne_state = flight_control.flight_state in (
        FlightState.TAKEOFF,
        FlightState.FLYING,
        FlightState.LANDING,
    )
    if not airborne_state or not bridge.control_is_stale(max_age_s):
        return False
    data.ctrl[bindings.rotor_thrust_actuator_ids] = 0.0
    flight_control.handle(UiCommand(UiCommandType.EMERGENCY_STOP))
    bridge.request_disarm()
    return True


class ManagedPassiveViewer:
    """Owns the passive render thread until native window teardown completes.

    MuJoCo 3.6's public launch_passive starts an unowned daemon thread, and
    Handle.close only signals exit. Its GLFW atexit callback can consequently
    race render_loop. Keep the installed launcher's setup, but own and join a
    non-daemon thread. The private API is isolated here for version review.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Starts the Linux viewer and waits for its handle or startup error.

        Args:
            model: Loaded MuJoCo model to render.
            data: Matching simulation state; no physics is advanced here.

        Raises:
            RuntimeError: If the launcher is unavailable or rendering fails.
        """
        if sys.platform == "darwin":
            raise RuntimeError("ManagedPassiveViewer requires Linux/Windows")
        launcher = getattr(mujoco.viewer, "_launch_internal", None)
        if not callable(launcher):
            raise RuntimeError("MuJoCo viewer launcher is unavailable")
        mujoco.mj_forward(model, data)
        ready: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._thread_error: BaseException | None = None

        def render() -> None:
            """Runs native setup/render/teardown and reports Python failures."""
            try:
                launcher(
                    model, data, run_physics_thread=False, handle_return=ready,
                )
            except BaseException as error:
                self._thread_error = error
                ready.put(error)

        self._thread = threading.Thread(
            target=render, name="px4-mujoco-viewer", daemon=False,
        )
        self._thread.start()
        result = ready.get()
        if isinstance(result, BaseException):
            self._thread.join()
            raise RuntimeError(f"MuJoCo viewer startup failed: {result}") from result
        self._handle = result

    def __getattr__(self, name: str) -> Any:
        """Forwards scene, lock and sync access to the native viewer handle."""
        return getattr(self._handle, name)

    def is_running(self) -> bool:
        """Reports whether the viewer remains open without a render failure."""
        return self._thread_error is None and self._handle.is_running()

    def close(self) -> None:
        """Requests close and waits for native teardown before returning."""
        if self._thread.is_alive():
            self._handle.close()
        self._thread.join()
        if self._thread_error is not None:
            raise RuntimeError(
                f"MuJoCo render thread failed: {self._thread_error}"
            ) from self._thread_error


class WindSimulationApp:
    """Coordinates MuJoCo, PX4, wind physics, UI, and diagnostics."""

    ROTOR_SPIN_DIRECTIONS = np.array([1.0, -1.0, -1.0, 1.0])
    ROTOR_VISUAL_SPEED_RAD_S = 180.0
    CONTROL_STALE_TIMEOUT_S = 0.5
    FIRST_CONTROL_TIMEOUT_S = 3.0
    UNLOCK_CONFIRM_TIMEOUT_S = 5.0
    PANEL_UPDATE_PERIOD_S = 0.2
    TARGET_YAW_RAD = 0.0

    def __init__(
        self,
        config: SimulationConfig,
        bridge: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        pinn_predictor: Any | None = None,
    ) -> None:
        """Loads model state and constructs non-graphical runtime components.

        Args:
            config: Validated simulation configuration.
            bridge: Optional bridge test double or preconstructed bridge.
            clock: Monotonic wall clock.
            sleep: Injectable sleep function used for real-time pacing.
            pinn_predictor: Optional residual-model test double.
        """
        if not config.xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {config.xml_path}")
        if not config.params_path.is_file():
            raise FileNotFoundError(f"PX4 params not found: {config.params_path}")
        self.config = config
        self.clock = clock
        self.sleep = sleep
        self.model = mujoco.MjModel.from_xml_path(str(config.xml_path))
        self.data = mujoco.MjData(self.model)
        self.bindings = ModelBindings.resolve(self.model)
        mujoco.mj_forward(self.model, self.data)

        initial_position = self.data.xpos[
            self.bindings.quadrotor_body_id
        ].copy()
        self.flight_control = FlightControlState(
            initial_position_world=initial_position,
            takeoff_height_m=config.takeoff_height_m,
        )
        self.wind_field: ConstantWindField | None = None
        if pinn_predictor is not None:
            self.pinn_predictor = pinn_predictor
        elif config.wind_backend == "analytic":
            self.pinn_predictor = AnalyticalWindResidualModel(
                config.reference_config, config.xml_path, self.model)
        elif config.wind_backend == "pinn-x4":
            self.pinn_predictor = X4PinnWindResidualModel(
                config.pinn_checkpoint, config.reference_config, config.xml_path,
                self.model)
        else:
            self.pinn_predictor = PinnWindResidualModel()
        self.residual_filter = ResidualSafetyFilter()
        hover_thrust = read_px4_param(
            config.params_path,
            "MPC_THR_HOVER",
            default=0.5,
        )
        total_max_thrust = compute_hover_matched_total_max_thrust(
            self.bindings.uav_mass_kg,
            hover_thrust,
        )
        rotor_scales = equal_rotor_thrust_scales(total_max_thrust)
        self.bridge = bridge or Px4Bridge(
            rotor_thrust_scales=rotor_scales,
            sim_tcp_port=config.sim_tcp_port,
            command_udp_port=config.command_udp_port,
            clock=clock,
        )
        if hasattr(self.bridge, "set_local_origin"):
            self.bridge.set_local_origin(initial_position)

        self.command_queue: queue.SimpleQueue[UiCommand] = queue.SimpleQueue()
        self.panel: ControlPanel | None = None
        self.viewer: Any | None = None
        self.active_trajectory: PredefinedXYTrajectory | None = None
        self.trajectory_plot: RealtimePredefinedTrajectoryPlot | None = None
        self.last_error = "none"
        self.last_active_wind = np.zeros(3, dtype=np.float64)
        self.last_relative_velocity = np.zeros(3, dtype=np.float64)
        self.last_raw_residual_acceleration = np.zeros(3, dtype=np.float64)
        self.last_filtered_residual_acceleration = np.zeros(
            3,
            dtype=np.float64,
        )
        self.last_pinn_force = np.zeros(3, dtype=np.float64)
        self.last_rotor_thrusts = np.zeros(4, dtype=np.float64)
        self.previous_velocity_world = np.zeros(3, dtype=np.float64)
        self.last_panel_update_time = -np.inf
        self.flight_started_wall_time = 0.0
        self.unlock_requested = False
        self.unlock_request_started_wall_time: float | None = None
        self.unlock_ack_accepted = False
        self.last_processed_arm_ack_sequence = 0
        self.exit_requested = False
        self._resolve_auxiliary_bindings()
        self._hold_arm_initial_pose()
        self.last_wind_compute_wall_s = 0.0
        self.run_logger = (
            WindRunLogger(config.run_log, self._run_metadata(
                predictor_overridden=pinn_predictor is not None,
            )) if config.run_log is not None else None
        )
        print(f"Wind backend: {config.wind_backend}; "
              "horizontal resultant <=3 m/s; XY EMA/limit unchanged.",
              flush=True)
        if pinn_predictor is None and config.wind_backend in ("analytic", "pinn-x4"):
            print("Frozen reference: "
                  f"{self.pinn_predictor.reference_sha256}; "
                  "fixed initial arm geometry, force-only increment.",
                  flush=True)
        if pinn_predictor is None and config.wind_backend == "pinn-x4":
            print(f"X4 PINN checkpoint: {self.pinn_predictor.checkpoint_sha256}; "
                  "CPU float32, one Torch thread; raw pair difference.", flush=True)

    def _resolve_auxiliary_bindings(self) -> None:
        """Resolves arm-hold actuators and required body IMU sensors."""
        self.arm_actuator_ids: list[int] = []
        self.arm_qpos_adrs: list[int] = []
        for joint_index in range(1, 7):
            name = f"joint{joint_index}"
            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if actuator_id < 0 or joint_id < 0:
                raise ValueError(f"missing required arm hold element: {name}")
            self.arm_actuator_ids.append(int(actuator_id))
            self.arm_qpos_adrs.append(int(self.model.jnt_qposadr[joint_id]))
        self.arm_hold_controls = self.data.qpos[self.arm_qpos_adrs].copy()
        self.body_gyro_sensor_id = self._required_sensor_id("body_gyro")
        self.body_linacc_sensor_id = self._required_sensor_id("body_linacc")

    def _required_sensor_id(self, name: str) -> int:
        """Returns a required sensor ID or raises a named error."""
        sensor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            name,
        )
        if sensor_id < 0:
            raise ValueError(f"missing required sensor:{name}")
        return int(sensor_id)

    def _sensor_vector(self, sensor_id: int) -> FloatArray:
        """Reads one three-dimensional MuJoCo sensor value."""
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        if dimension != 3:
            raise ValueError(f"sensor {sensor_id} must have dimension 3")
        return self.data.sensordata[address : address + dimension].copy()

    def _hold_arm_initial_pose(self) -> None:
        """Keeps all six arm position actuators at their initial joint values."""
        self.data.ctrl[self.arm_actuator_ids] = self.arm_hold_controls

    def _arm_joint_deviations(self) -> dict[str, float]:
        """Return actual minus qpos0 for six arm and two gripper joints."""
        result = {}
        for name in (*[f"joint{i}" for i in range(1, 7)],
                     "endleft", "endright"):
            address = int(self.model.jnt_qposadr[self.model.joint(name).id])
            result[name] = float(
                self.data.qpos[address] - self.model.qpos0[address]
            )
        return result

    def _run_metadata(self, predictor_overridden: bool) -> dict[str, Any]:
        """Describe actual backend, source hashes and shared run settings."""
        files = [self.config.xml_path, self.config.params_path,
                 Path(__file__).resolve(),
                 Path(__file__).with_name("run_logging.py")]
        if self.config.reference_config is not None and not predictor_overridden:
            files.append(self.config.reference_config)
            files.extend(Path(__file__).parent / name for name in (
                "wind_backends.py", "training/reference_aerodynamics.py",
                "training/xml_parameters.py",
            ))
            if self.config.pinn_checkpoint is not None:
                files.append(self.config.pinn_checkpoint)
                files.extend(Path(__file__).parent / name for name in (
                    "training/model_v1.py", "training/sampling.py",
                    "training/dataset_v1.py",
                ))
        elif not predictor_overridden:
            files.extend(default_pinn_checkpoint_dir() / name for name in (
                "best_model.pt", "normalizer.pt",
            ))
        return {
            "wind_backend": self.config.wind_backend,
            "predictor_type": type(self.pinn_predictor).__name__,
            "predictor_overridden": predictor_overridden,
            "reference_sha256": getattr(
                self.pinn_predictor, "reference_sha256", None,
            ),
            **{name: getattr(self.pinn_predictor, name, None) for name in (
                "checkpoint_sha256", "dataset_manifest_sha256", "dataset_archive_sha256",
                "inference_device", "inference_dtype", "torch_threads",
            )},
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            },
            "mujoco": mujoco.__version__, "numpy": np.__version__,
            "mass_total_kg": self.bindings.uav_mass_kg,
            "timestep_s": float(self.model.opt.timestep),
            "ema_alpha": self.residual_filter.ema_alpha,
            "acceleration_limit_m_s2": (
                self.residual_filter.acceleration_limit_m_s2
            ),
            "wind_max_m_s": 3.0,
            "takeoff_height_m": self.config.takeoff_height_m,
            "trajectory_size_m": self.config.trajectory_size_m,
            "trajectory_duration_s": self.config.trajectory_duration_s,
            "figure8_duration_s": self.config.figure8_duration_s,
            "barometer_seed": BAROMETER_RANDOM_SEED,
            "joint_deviation_units": "joint1-6 rad; endleft/endright m",
        }

    def _run_log_sample(self) -> dict[str, Any] | None:
        """Capture synchronized state and applied force before physics step."""
        if self.run_logger is None or not self.run_logger.due(self.data.time):
            return None
        _, velocity, quaternion, omega = self._current_state()
        address = self.bindings.main_qpos_adr
        # mj_step advances qpos/qvel but leaves derived caches at the prior
        # evaluation instant. Observe integrated state without extra dynamics.
        position = self.data.qpos[address:address + 3].copy()
        rotation_buffer = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_buffer, quaternion)
        rotation = rotation_buffer.reshape(3, 3)
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        outside = bool(
            np.hypot(*velocity[:2]) > 3 or abs(velocity[2]) > 1
            or abs(pitch) > np.deg2rad(20)
            or abs(roll) > np.deg2rad(20) or np.linalg.norm(omega) > 1.5
        )
        return {
            "sim_time_s": float(self.data.time),
            "position_world_m": position.tolist(),
            "velocity_world_m_s": velocity.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "angular_velocity_body_rad_s": omega.tolist(),
            "target_world_m": self.flight_control.target_position_world.tolist(),
            "trajectory": self.flight_control.active_trajectory_name,
            "flight_state": self.flight_control.flight_state.name,
            "wind_enabled": self.flight_control.wind_enabled,
            "configured_wind_world_m_s": (
                self.wind_field.configured_velocity_world.tolist()
                if self.wind_field is not None else None
            ),
            "active_wind_world_m_s": self.last_active_wind.tolist(),
            "raw_acceleration_world_m_s2": (
                self.last_raw_residual_acceleration.tolist()
            ),
            "filtered_acceleration_world_m_s2": (
                self.last_filtered_residual_acceleration.tolist()
            ),
            "force_world_n": self.last_pinn_force.tolist(),
            "arm_joint_deviation": self._arm_joint_deviations(),
            "outside_inspection_domain": outside,
            "acceleration_clipped": bool(np.any(
                np.abs(self.residual_filter.filtered[:2])
                > self.residual_filter.acceleration_limit_m_s2
            )),
            "rotor_thrust_n": self.last_rotor_thrusts.tolist(),
            "wind_compute_wall_s": self.last_wind_compute_wall_s,
            "last_error": self.last_error,
        }

    def _current_state(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Returns position, velocity, quaternion, and body angular rate."""
        qpos_address = self.bindings.main_qpos_adr
        dof_address = self.bindings.main_dof_adr
        position = self.data.xpos[
            self.bindings.quadrotor_body_id
        ].copy()
        velocity = self.data.qvel[dof_address : dof_address + 3].copy()
        quaternion = self.data.qpos[
            qpos_address + 3 : qpos_address + 7
        ].copy()
        angular_rate = self.data.qvel[
            dof_address + 3 : dof_address + 6
        ].copy()
        for name, vector in (
            ("position", position),
            ("velocity", velocity),
            ("quaternion", quaternion),
            ("angular_rate", angular_rate),
        ):
            if not np.all(np.isfinite(vector)):
                raise FloatingPointError(f"non-finite MuJoCo {name}")
        return position, velocity, quaternion, angular_rate

    def _zero_rotors(self) -> None:
        """Commands zero thrust at the four resolved rotor actuators."""
        self.data.ctrl[self.bindings.rotor_thrust_actuator_ids] = 0.0
        self.last_rotor_thrusts = np.zeros(4, dtype=np.float64)

    def _finish_unlock_request(self, error: str = "none") -> None:
        """Ends the pending request and exposes any failure to the user.

        Args:
            error: Failure description, or ``none`` after confirmed arming.
        """
        self.unlock_requested = False
        self.unlock_request_started_wall_time = None
        self.unlock_ack_accepted = False
        self.last_error = error
        if error != "none":
            print(error, flush=True)

    def _update_unlock_request_state(self) -> None:
        """Consumes fresh arm ACK events and bounds heartbeat confirmation."""
        if self.flight_control.armed:
            if self.unlock_requested:
                self._finish_unlock_request()
            return
        if not self.unlock_requested:
            return
        sequence = self.bridge.arm_ack_sequence
        if sequence != self.last_processed_arm_ack_sequence:
            self.last_processed_arm_ack_sequence = sequence
            result = self.bridge.last_arm_ack_result
            if result in (
                mavutil.mavlink.MAV_RESULT_ACCEPTED,
                mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
            ):
                self.unlock_ack_accepted = True
            else:
                status = self.bridge.last_statustext
                if status in ("", "none"):
                    status = "no PX4 STATUSTEXT received yet"
                entry = mavutil.mavlink.enums["MAV_RESULT"].get(result)
                name = entry.name if entry is not None else "UNKNOWN"
                self._finish_unlock_request(
                    f"PX4 arm rejected (MAV_RESULT={result}, {name}): {status}"
                )
                return
        started = self.unlock_request_started_wall_time
        if started is not None and (
            self.clock() - started >= self.UNLOCK_CONFIRM_TIMEOUT_S
        ):
            reason = (
                "ARM ACK received, but no armed heartbeat"
                if self.unlock_ack_accepted else "no ARM ACK received"
            )
            self._finish_unlock_request(
                f"PX4 unlock timed out: {reason}. "
                f"Last PX4 status: {self.bridge.last_statustext}"
            )

    def _process_ui_command(self, command: UiCommand) -> None:
        """Runs state-machine actions for one queued UI command."""
        if command.command_type is UiCommandType.EXIT:
            if self.exit_requested:
                return
            self.exit_requested = True
            if self.unlock_requested:
                self._finish_unlock_request()
            self._zero_rotors()
            self.bridge.request_disarm()
            self.bridge.request_shutdown()
            return
        if command.command_type is UiCommandType.SET_WIND:
            try:
                wind_field = ConstantWindField(command.payload)
            except (TypeError, ValueError, OverflowError) as error:
                print(f"Wind settings rejected: {error}", flush=True)
                return
            self.wind_field = wind_field
            self.flight_control.wind_configured = True
            enabled = "ON" if self.flight_control.wind_enabled else "OFF"
            print(
                "Wind velocity updated (world): "
                f"{wind_field.configured_velocity_world.tolist()} m/s; Wind {enabled}",
                flush=True,
            )
            return
        actions = self.flight_control.handle(command)
        for action in actions:
            if action is ControlAction.REQUEST_ARM:
                if self.unlock_requested:
                    continue
                self.unlock_requested = True
                self.unlock_request_started_wall_time = self.clock()
                self.unlock_ack_accepted = False
                self.last_processed_arm_ack_sequence = self.bridge.arm_ack_sequence
                self.last_error = "none"
                print("PX4 unlock requested; waiting for confirmation", flush=True)
                self.bridge.request_arm()
            elif action is ControlAction.REQUEST_DISARM:
                if self.unlock_requested:
                    self._finish_unlock_request()
                self.bridge.request_disarm()
            elif action is ControlAction.ENTER_OFFBOARD:
                self.flight_started_wall_time = self.clock()
                self.bridge.request_offboard(force=True)
            elif action in (
                ControlAction.SEND_TAKEOFF_TARGET,
                ControlAction.SEND_LANDING_TARGET,
                ControlAction.SEND_POSITION_TARGET,
            ):
                self.bridge.send_position_target(
                    self.flight_control.target_position_world,
                    self.TARGET_YAW_RAD,
                )
            elif action is ControlAction.START_TRAJECTORY:
                trajectory_name = self.flight_control.active_trajectory_name
                duration = (
                    self.config.figure8_duration_s
                    if trajectory_name == "figure8"
                    else self.config.trajectory_duration_s
                )
                self.active_trajectory = PredefinedXYTrajectory.from_start(
                    name=trajectory_name,
                    start_xy=self.flight_control.target_position_world[:2],
                    altitude=float(self.flight_control.target_position_world[2]),
                    start_time=float(self.data.time),
                    duration=duration,
                    size_m=self.config.trajectory_size_m,
                )
                if self.trajectory_plot is not None:
                    self.trajectory_plot.close()
                self.trajectory_plot = (
                    create_realtime_predefined_trajectory_plot(
                        self.active_trajectory
                    )
                )
                show_realtime_figure(self.trajectory_plot.figure)
            elif action is ControlAction.ZERO_THRUST:
                self._zero_rotors()
        if (
            command.command_type
            in (UiCommandType.LAND, UiCommandType.EMERGENCY_STOP)
            and self.flight_control.active_trajectory_name == "none"
        ):
            self.active_trajectory = None

    def _drain_ui_commands(self) -> None:
        """Processes all currently queued panel commands."""
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            self._process_ui_command(command)

    def _update_trajectory_target(self) -> None:
        """Samples or finishes the active reference trajectory."""
        if self.active_trajectory is None:
            return
        sample = self.active_trajectory.sample(float(self.data.time))
        if sample is None:
            self.active_trajectory = None
            self.flight_control.active_trajectory_name = "none"
            return
        self.flight_control.target_position_world = sample.position_world.copy()

    def _apply_wind_force(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Apply either backend through the same world-XY filter/force path."""
        _position, velocity, quaternion, angular_velocity_body = (
            self._current_state()
        )
        # Both predictors use current free-joint body rate, consistent with
        # current qpos/qvel, rather than the previous step's gyro cache.
        wind_enabled = (
            self.flight_control.wind_enabled and self.wind_field is not None
        )
        active_wind = (
            self.wind_field.velocity(wind_enabled)
            if self.wind_field is not None else np.zeros(3, dtype=np.float64)
        )
        relative_velocity = velocity - active_wind

        if not wind_enabled:
            self.residual_filter.reset()
            raw_residual = np.zeros(3, dtype=np.float64)
            filtered_residual = np.zeros(3, dtype=np.float64)
            force = np.zeros(3, dtype=np.float64)
        else:
            raw_residual = self.pinn_predictor.predict(
                velocity_world=velocity,
                quaternion_wxyz=quaternion,
                angular_velocity_body=angular_velocity_body,
                wind_world=active_wind,
            )
            # Horizontal-only disturbance: P = diag(1, 1, 0). Preserve the
            # original 3D prediction for diagnostics, but do not apply its Z
            # component. Clear previous vertical EMA state without resetting XY.
            horizontal_residual = raw_residual.copy()
            horizontal_residual[2] = 0.0
            self.residual_filter.filtered[2] = 0.0
            filtered_residual = self.residual_filter.update(horizontal_residual)
            force = self.bindings.uav_mass_kg * filtered_residual

        apply_external_wind_force(
            self.data,
            self.bindings.quadrotor_body_id,
            force,
        )
        self.last_active_wind = active_wind.copy()
        self.last_relative_velocity = relative_velocity.copy()
        self.last_raw_residual_acceleration = raw_residual.copy()
        self.last_filtered_residual_acceleration = filtered_residual.copy()
        self.last_pinn_force = force.copy()
        return (
            active_wind.copy(),
            relative_velocity.copy(),
            force.copy(),
        )

    def _apply_px4_rotor_control(self) -> None:
        """Maps fresh PX4 output to named rotor actuators when armed."""
        thrusts = self.bridge.receive_rotor_thrusts()
        thrusts = np.asarray(thrusts, dtype=np.float64)
        if thrusts.shape != (4,) or not np.all(np.isfinite(thrusts)):
            raise FloatingPointError("PX4 rotor thrusts must be finite with shape (4,)")
        if (
            not self.flight_control.armed
            or self.flight_control.flight_state is FlightState.EMERGENCY_STOP
        ):
            self._zero_rotors()
            return
        self.data.ctrl[self.bindings.rotor_thrust_actuator_ids] = thrusts
        self.last_rotor_thrusts = thrusts.copy()

    def _update_visual_rotors(self) -> None:
        """Spins propeller meshes whenever physical thrust is nonzero."""
        if np.any(self.last_rotor_thrusts > 1e-4):
            spin_controls = (
                self.ROTOR_SPIN_DIRECTIONS * self.ROTOR_VISUAL_SPEED_RAD_S
            )
        else:
            spin_controls = np.zeros(4, dtype=np.float64)
        self.data.ctrl[self.bindings.rotor_spin_actuator_ids] = spin_controls

    def _panel_snapshot(self, position_world: FloatArray) -> PanelSnapshot:
        """Builds an immutable status snapshot for the panel."""
        return PanelSnapshot(
            px4_connected=bool(
                getattr(self.bridge, "connected", False)
                and (
                    not hasattr(self.bridge, "heartbeat_is_recent")
                    or self.bridge.heartbeat_is_recent()
                )
            ),
            armed=self.flight_control.armed,
            offboard=self.flight_control.offboard,
            flight_state=self.flight_control.flight_state.name,
            trajectory=self.flight_control.active_trajectory_name,
            position_world=position_world,
            target_world=self.flight_control.target_position_world.copy(),
            configured_wind_world=(
                self.wind_field.configured_velocity_world.copy()
                if self.wind_field is not None else None
            ),
            active_wind_world=self.last_active_wind,
            filtered_residual_acceleration=(
                self.last_filtered_residual_acceleration
            ),
            pinn_force_world=self.last_pinn_force,
            wind_enabled=self.flight_control.wind_enabled,
            last_error=self.last_error,
            unlock_pending=(
                self.unlock_requested and not self.flight_control.armed
            ),
            wind_backend=self.config.wind_backend,
            simulation_time_s=float(self.data.time),
        )

    def _run_iteration(self) -> bool:
        """Runs one ordered real-time MuJoCo/PX4 physics iteration.

        Returns:
            ``False`` when Exit stops the iteration, otherwise ``True``.
        """
        iteration_start = time.perf_counter()
        position, velocity, quaternion, angular_rate = self._current_state()
        acceleration = (
            velocity - self.previous_velocity_world
        ) / float(self.model.opt.timestep)
        self.previous_velocity_world = velocity.copy()
        specific_force_body = self._sensor_vector(self.body_linacc_sensor_id)
        gyro_body = self._sensor_vector(self.body_gyro_sensor_id)
        if np.all(np.isfinite(gyro_body)):
            angular_rate = gyro_body
        self.bridge.send_state(
            position_world=position,
            velocity_world=velocity,
            acceleration_world=acceleration,
            quaternion_wxyz=quaternion,
            angular_rate_body=angular_rate,
            specific_force_body=specific_force_body,
        )
        self.bridge.send_command_heartbeat()
        self.bridge.poll_status()

        previous_state = self.flight_control.flight_state
        self.flight_control.update_px4_status(
            armed=bool(self.bridge.armed),
            offboard=bool(self.bridge.offboard),
        )
        self.flight_control.update_position(position)
        self._update_unlock_request_state()
        if self.unlock_requested and not self.unlock_ack_accepted:
            self.bridge.request_arm()
        if (
            previous_state is FlightState.LANDING
            and self.flight_control.flight_state is FlightState.ARMED
        ):
            self.bridge.request_disarm()

        self._drain_ui_commands()
        if self.exit_requested:
            return False
        self._update_trajectory_target()
        if self.active_trajectory is not None and self.trajectory_plot is not None:
            update_realtime_predefined_trajectory_plot(
                self.trajectory_plot,
                position_world=position,
                sim_time=float(self.data.time),
            )
        if self.flight_control.flight_state in (
            FlightState.TAKEOFF,
            FlightState.FLYING,
            FlightState.LANDING,
        ):
            self.bridge.send_position_target(
                self.flight_control.target_position_world,
                self.TARGET_YAW_RAD,
            )
            if not self.flight_control.offboard:
                self.bridge.request_offboard()

        self._apply_px4_rotor_control()
        stale_triggered = enforce_stale_control_safety(
            data=self.data,
            bindings=self.bindings,
            flight_control=self.flight_control,
            bridge=self.bridge,
            max_age_s=self.CONTROL_STALE_TIMEOUT_S,
        )
        if stale_triggered:
            self.last_error = "PX4 actuator stream stale; emergency stop latched"
        elif (
            self.flight_control.flight_state is FlightState.TAKEOFF
            and self.flight_started_wall_time > 0.0
            and getattr(self.bridge, "last_control_received_wall_time", 0.0) <= 0.0
            and self.clock() - self.flight_started_wall_time
            > self.FIRST_CONTROL_TIMEOUT_S
        ):
            self._process_ui_command(
                UiCommand(UiCommandType.EMERGENCY_STOP)
            )
            self.last_error = "PX4 supplied no actuator control after takeoff"

        self._hold_arm_initial_pose()
        self._update_visual_rotors()
        wind_start = time.perf_counter()
        self._apply_wind_force()
        self.last_wind_compute_wall_s = time.perf_counter() - wind_start
        sample = self._run_log_sample()
        mujoco.mj_step(self.model, self.data)
        if sample is not None:
            sample["iteration_wall_s"] = time.perf_counter() - iteration_start
            self.run_logger.record(sample)

        if (
            self.panel is not None
            and self.data.time - self.last_panel_update_time
            >= self.PANEL_UPDATE_PERIOD_S
        ):
            self.last_panel_update_time = float(self.data.time)
            self.panel.update(self._panel_snapshot(position))
        return True

    def _runtime_is_open(self) -> bool:
        """Returns whether all enabled runtime windows remain open."""
        panel_open = self.panel is not None and self.panel.is_open
        viewer_open = (
            not self.config.viewer_enabled
            or (self.viewer is not None and self.viewer.is_running())
        )
        return bool(panel_open and viewer_open)

    def _close_runtime(self) -> None:
        """Always joins rendering, including after transport/reset failures."""
        active_exception = sys.exc_info()[1]

        def close_log(
            error_type: type[BaseException] | None,
            error: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            """Record errors from the run and any later cleanup callbacks."""
            del error_type, traceback
            failure = error if error is not None else active_exception
            self.run_logger.close({
                "exit_requested": self.exit_requested,
                "exception": repr(failure) if failure is not None else None,
                "flight_state": self.flight_control.flight_state.name,
                "last_error": self.last_error,
            })
            return False

        with ExitStack() as cleanup:
            if self.run_logger is not None:
                cleanup.push(close_log)
            if hasattr(self.bridge, "close"):
                cleanup.callback(self.bridge.close)
            if self.panel is not None:
                cleanup.callback(self.panel.close)
            if self.trajectory_plot is not None:
                cleanup.callback(self.trajectory_plot.close)
            if self.viewer is not None:
                cleanup.callback(self.viewer.close)
            self._zero_rotors()
            self._hold_arm_initial_pose()
            apply_external_wind_force(
                self.data,
                self.bindings.quadrotor_body_id,
                np.zeros(3, dtype=np.float64),
            )
            mujoco.mj_forward(self.model, self.data)
            if (
                getattr(self.bridge, "connected", False)
                and not self.exit_requested
            ):
                self.bridge.request_disarm()

    def run(self) -> None:
        """Connects PX4 and runs the interactive real-time experiment."""
        try:
            self.panel = ControlPanel(
                self.command_queue,
                initial_wind_world=self.config.wind_velocity_world,
            )
            plt.ion()
            self.panel.figure.show()
            if self.config.viewer_enabled:
                self.viewer = ManagedPassiveViewer(self.model, self.data)
            if not getattr(self.bridge, "connected", False):
                self.bridge.connect()
            wall_origin = self.clock() - float(self.data.time)
            while self._runtime_is_open():
                if not self._run_iteration():
                    break
                if self.viewer is not None:
                    with self.viewer.lock():
                        draw_wind_arrows(
                            self.viewer.user_scn,
                            self.last_active_wind,
                            float(self.data.time),
                        )
                    self.viewer.sync()
                next_step_wall_time = (
                    wall_origin
                    + float(self.data.time)
                    + float(self.model.opt.timestep)
                )
                remaining = next_step_wall_time - self.clock()
                if remaining > 0.0:
                    self.sleep(min(remaining, float(self.model.opt.timestep)))
        finally:
            self._close_runtime()


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the standalone PX4/MuJoCo wind simulation."""
    # SIGSEGV bypasses Python exceptions; retain all thread stacks on stderr.
    faulthandler.enable(all_threads=True)
    try:
        config = parse_config(argv)
        wind_draft = (
            config.wind_velocity_world.tolist()
            if config.wind_velocity_world is not None else None
        )
        print(
            "PX4 MuJoCo wind configuration: "
            f"xml={config.xml_path}, params={config.params_path}, "
            f"wind=unconfigured (Apply Wind required), draft={wind_draft}, "
            f"backend={config.wind_backend}, "
            f"ports={config.sim_tcp_port}/{config.command_udp_port}",
            flush=True,
        )
        print(
            f"Runtime diagnostics: MuJoCo={mujoco.__version__}, "
            f"Matplotlib backend={plt.get_backend()}, faulthandler=enabled",
            flush=True,
        )
        app = WindSimulationApp(config)
        try:
            app.run()
        finally:
            if app.exit_requested:
                state_directory = Path(
                    os.environ.get("PX4_STATE_DIR", repository_root() / "outputs/px4")
                )
                if request_managed_sitl_shutdown(state_directory):
                    print("Local PX4 SITL shutdown requested.", flush=True)
                else:
                    print(
                        "No managed local SITL found. If PX4 is still running, "
                        "stop it in terminal one; restart with scripts/run_px4_sitl.sh.",
                        flush=True,
                    )
    except (
        FileNotFoundError,
        ValueError,
        FloatingPointError,
        OSError,
        RuntimeError,
    ) as error:
        print(f"PX4 wind simulation error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("PX4 wind simulation interrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
