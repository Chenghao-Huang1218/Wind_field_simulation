"""Freeze default-scene facts and explicitly selected aerodynamic proxies.

All kinematics use scratch MjData at qpos0; no stepping or runtime edits.
MuJoCo compiled geom transforms already account for mesh centering, scaling
and principal-axis rotation. See MuJoCo 3.6 MJCF mesh documentation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = (
    PROJECT_ROOT / "src/wind_field_simulation/assets/scenes/default"
    / "robot_quadrotor_950_with_arm.xml"
)
DEFAULT_ASSUMPTIONS = PROJECT_ROOT / "configs/pinn/x4_950_reference_v1.json"
ARM_BODIES = (
    "arm_base", "link1", "link2", "link3", "link4", "link5", "link6",
    "left", "right",
)


def content_sha256(value: Any) -> str:
    """Return a deterministic JSON fingerprint (excluding no fields)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def xml_dependency_closure(xml_path: Path) -> list[Path]:
    """Collect recursive includes relative to the main MJCF directory.

    Args:
        xml_path: Main scene path, resolved before loading.

    Returns:
        Sorted unique paths including the main scene and every actual include.
    """
    # MJCF 3.6 include/file is relative to the MAIN model directory, even for
    # nested includes (XMLreference.html#include). Mesh paths instead come
    # from the compiled model, which preserves their resolved asset paths.
    scene = xml_path.resolve()
    paths: set[Path] = set()
    pending = [scene]
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        for include in ET.parse(path).getroot().iter("include"):
            pending.append((scene.parent / include.attrib["file"]).resolve())
    return sorted(paths)


def _label(path: Path) -> str:
    """Use relocatable project paths, absolute paths only for external inputs."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _rotation(quaternion: FloatArray) -> FloatArray:
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quaternion)
    return result.reshape(3, 3)


def _semiaxes(model: mujoco.MjModel, geom_id: int) -> FloatArray:
    """Convert a primitive's extents into an equivalent ellipsoid's radii."""
    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return size.copy()
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.repeat(size[0], 3)
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return size.copy()
    if geom_type in (
        mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE,
    ):
        cap = size[0] if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
        return np.array([size[0], size[0], size[1] + cap])
    raise ValueError(f"Unsupported primitive: geom {geom_id}, type {geom_type}")


def _body_points(model: mujoco.MjModel, geom_id: int) -> FloatArray:
    """Return actual mesh vertices or primitive bound corners in owner frame."""
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        end = start + model.mesh_vertnum[mesh_id]
        points = model.mesh_vert[start:end].astype(float)
    else:
        corners = np.array(list(itertools.product((-1, 1), repeat=3)))
        points = corners * _semiaxes(model, geom_id)
    return points @ _rotation(model.geom_quat[geom_id]).T + model.geom_pos[geom_id]


def _geometry_manifest(model: mujoco.MjModel, data: mujoco.MjData,
                       root_id: int, body_ids: list[int]) -> dict[str, Any]:
    """Select each source once; excluded sources receive explicit reasons."""
    root_rotation = data.xmat[root_id].reshape(3, 3)
    root_position = data.xpos[root_id]
    selected: set[int] = set()
    proxies: list[dict[str, Any]] = []

    def add_proxy(
        name: str, body_id: int, geom_ids: list[int], method: str,
    ) -> None:
        body_rotation = root_rotation.T @ data.xmat[body_id].reshape(3, 3)
        body_position = root_rotation.T @ (data.xpos[body_id] - root_position)
        if method == "collision_primitive":
            geom_id = geom_ids[0]
            radii = _semiaxes(model, geom_id)
            center = body_position + body_rotation @ model.geom_pos[geom_id]
            orientation = body_rotation @ _rotation(model.geom_quat[geom_id])
        else:
            points = np.concatenate([
                _body_points(model, geom_id) for geom_id in geom_ids
            ])
            lower, upper = points.min(axis=0), points.max(axis=0)
            radii = (upper - lower) / 2
            center = body_position + body_rotation @ ((upper + lower) / 2)
            orientation = body_rotation
        if np.any(radii <= 0):
            raise ValueError(f"Degenerate proxy {name}")
        sources = []
        for geom_id in geom_ids:
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = model.mesh(mesh_id).name if mesh_id >= 0 else None
            sources.append({
                "geom_id": geom_id, "geom_name": model.geom(geom_id).name,
                "mesh_name": mesh_name,
            })
        proxies.append({
            "id": name, "mass_owner": model.body(body_id).name,
            "mass_note": (
                "Owner mass is recorded once in bodies; proxy adds no mass."
            ),
            "method": method, "sources": sources, "semiaxes_m": radii.tolist(),
            "center_body_m": center.tolist(),
            "rotation_body_from_proxy": orientation.tolist(),
            "areas_m2": (np.pi * np.prod(radii) / radii).tolist(),
            "selection_reason": (
                "One aircraft collision primitive; visual mesh excluded."
                if method == "collision_primitive" else
                "One union AABB per arm body; collision preferred, "
                "visual fallback only without collision."
            ),
        })
        selected.update(geom_ids)

    for geom_id in range(model.ngeom):
        if model.geom(geom_id).name.startswith(("collision-", "chassis-")):
            if int(model.geom_bodyid[geom_id]) not in body_ids:
                raise ValueError("Aircraft proxy outside vehicle subtree")
            add_proxy(model.geom(geom_id).name, int(model.geom_bodyid[geom_id]),
                      [geom_id], "collision_primitive")
    for name in ARM_BODIES:
        body_id = model.body(name).id
        geoms = [int(index) for index in np.flatnonzero(
            model.geom_bodyid == body_id
        )]
        collision = [index for index in geoms
                     if model.geom_contype[index] or model.geom_conaffinity[index]]
        chosen = collision or geoms
        if not chosen:
            raise ValueError(f"No geometry for {name}")
        method = ("collision_union_body_aabb" if collision
                  else "visual_mesh_body_aabb")
        add_proxy(name, body_id, chosen, method)
    excluded = []
    for geom_id in range(model.ngeom):
        if geom_id in selected:
            continue
        owner = model.body(int(model.geom_bodyid[geom_id])).name
        name = model.geom(geom_id).name
        if int(model.geom_bodyid[geom_id]) not in body_ids:
            reason = "outside vehicle subtree (scene/floor)"
        elif name.startswith("thruster-"):
            reason = "collision-only rotor disk; not a solid aerodynamic disk"
        elif owner.startswith("rotor-"):
            reason = "visual rotating blade; blade aerodynamics not modeled"
        elif owner == "quadrotor_950" or owner in ARM_BODIES:
            reason = "visual duplicate of selected collision/body proxy"
        else:
            raise ValueError(f"Unreviewed geom {geom_id} in body {owner}")
        excluded.append({"geom_id": geom_id, "geom_name": name,
                         "body_name": owner, "reason": reason})
    return {"proxies": proxies, "excluded_geoms": excluded,
            "non_geom_exclusions": ["sites", "cameras", "lights"],
            "limitations": (
                "No shielding/occlusion; junction overlap and collision "
                "geometry approximations remain."
            )}


def export_vehicle_parameters(xml_path: Path) -> dict[str, Any]:
    """Export default X4-950 assembly facts and reviewed initial-pose proxies.

    Args:
        xml_path: Scene XML; absolute path is used for MuJoCo loading.

    Returns:
        JSON-compatible parameters, geometry manifest and source fingerprints.

    Raises:
        ValueError: Required bindings or reviewed geometry are unavailable.
    """
    scene = xml_path.resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root_id = model.body("quadrotor_950_pos").id
    body_ids = []
    for body_id in range(1, model.nbody):
        ancestor = body_id
        while ancestor != root_id and ancestor != 0:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root_id:
            body_ids.append(body_id)
    rotation = data.xmat[root_id].reshape(3, 3)
    com = rotation.T @ (data.subtree_com[root_id] - data.xpos[root_id])
    inertia = np.zeros((3, 3))
    bodies = []
    for body_id in body_ids:
        position = rotation.T @ (data.xipos[body_id] - data.xpos[root_id])
        inertial_rotation = rotation.T @ data.ximat[body_id].reshape(3, 3)
        body_rotation = rotation.T @ data.xmat[body_id].reshape(3, 3)
        body_inertia = (
            inertial_rotation @ np.diag(model.body_inertia[body_id])
            @ inertial_rotation.T
        )
        displacement = position - com
        mass = float(model.body_mass[body_id])
        inertia += body_inertia + mass * (displacement @ displacement * np.eye(3)
                                         - np.outer(displacement, displacement))
        body_origin = rotation.T @ (data.xpos[body_id] - data.xpos[root_id])
        bodies.append({"name": model.body(body_id).name, "mass_kg": mass,
                       "com_body_m": position.tolist(),
                       "origin_body_m": body_origin.tolist(),
                       "rotation_root_from_body": body_rotation.tolist(),
                       "inertia_at_own_com_root_axes_kg_m2": body_inertia.tolist()})
    joints = {}
    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) not in body_ids:
            continue
        start = int(model.jnt_qposadr[joint_id])
        end = (int(model.jnt_qposadr[joint_id + 1])
               if joint_id + 1 < model.njnt else model.nq)
        joints[model.joint(joint_id).name] = {"qpos": data.qpos[start:end].tolist(),
                                            "type": int(model.jnt_type[joint_id])}
    actuators = []
    rotors = []
    for actuator_id in range(model.nu):
        item = {"name": model.actuator(actuator_id).name,
                "transmission_type": int(model.actuator_trntype[actuator_id]),
                "transmission_ids": model.actuator_trnid[actuator_id].tolist(),
                "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
                "gear": model.actuator_gear[actuator_id].tolist()}
        actuators.append(item)
        if item["name"] in ("rotor0", "rotor1", "rotor2", "rotor3"):
            site_id = int(model.actuator_trnid[actuator_id, 0])
            site_position = rotation.T @ (
                data.site_xpos[site_id] - data.xpos[root_id]
            )
            site_rotation = rotation.T @ data.site_xmat[site_id].reshape(3, 3)
            rotors.append({
                **item, "site_name": model.site(site_id).name,
                "position_body_m": site_position.tolist(),
                "rotation_body_from_site": site_rotation.tolist(),
            })
    xml_paths = xml_dependency_closure(scene)
    mesh_paths = set()
    for mesh_id in set(int(model.geom_dataid[index]) for index in range(model.ngeom)
                       if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH):
        address = int(model.mesh_pathadr[mesh_id])
        path = Path(model.paths[address:].split(b"\0", 1)[0].decode()).resolve()
        mesh_paths.add(path)
    dependencies = {}
    for group, paths in (
        ("xml_sha256", xml_paths), ("mesh_sha256", sorted(mesh_paths)),
    ):
        dependencies[group] = {
            _label(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    # Keep parameter-file mapping separate from the XML actuator hard limit.
    params_path = (
        PROJECT_ROOT / "src/wind_field_simulation/assets/configs/X4-950.params"
    )
    hover = None
    for line in params_path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[2] == "MPC_THR_HOVER":
            hover = float(fields[3])
    if hover is None or hover <= 0:
        raise ValueError("Missing valid MPC_THR_HOVER")
    gravity = float(np.linalg.norm(model.opt.gravity))
    mass_total = float(model.body_subtreemass[root_id])
    return {
        "kind": "x4_950_vehicle_parameters_v1", "scene": _label(scene),
        "mujoco_version": mujoco.__version__,
        "coordinates": {"root": "quadrotor_950_pos O (not CoM), FLU",
                        "world": "existing ENU-like world",
                        "quaternion": "WXYZ body to world",
                        "units": "SI"},
        "mass_total_kg": mass_total,
        "base_body_mass_kg": float(
            model.body_mass[model.body("quadrotor_950").id]
        ),
        "com_body_m": com.tolist(), "inertia_com_body_kg_m2": inertia.tolist(),
        "bodies": bodies, "initial_joints": joints,
        "actuators": actuators, "rotors": rotors,
        "options": {
            "timestep_s": float(model.opt.timestep),
            "gravity_world_m_s2": model.opt.gravity.tolist(),
            "density_kg_m3": float(model.opt.density),
            "viscosity_pa_s": float(model.opt.viscosity),
        },
        "bridge_thrust_mapping": {
            "source": _label(params_path),
            "sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
            "MPC_THR_HOVER": hover,
            "full_scale_per_motor_n": mass_total * gravity / hover / 4,
            "rpm": "unknown; visual spin is not calibrated RPM",
        },
        "dependencies": dependencies,
        "dependency_sha256": content_sha256(dependencies),
        "geometry_manifest": _geometry_manifest(model, data, root_id, body_ids),
    }


def export_reference_bundle(
    xml_path: Path, config_path: Path, output_dir: Path,
) -> dict[str, Any]:
    """Write vehicle, geometry and self-contained reference JSON artifacts.

    Args:
        xml_path: Default scene or a compatible reviewed X4-950 scene.
        config_path: Explicit aerodynamic assumptions, separate from XML facts.
        output_dir: Destination directory for three versioned artifacts.

    Returns:
        Self-contained reference config consumed by the analytical force model.
    """
    vehicle = export_vehicle_parameters(xml_path)
    manifest = vehicle.pop("geometry_manifest")
    config = json.loads(config_path.read_text())
    reference = {
        **config, "mass_total_kg": vehicle["mass_total_kg"],
        "proxies": manifest["proxies"],
        "initial_joints": vehicle["initial_joints"],
        "vehicle_sha256": content_sha256(vehicle),
        "geometry_sha256": content_sha256(manifest),
        "dependency_sha256": vehicle["dependency_sha256"],
        "assumptions_file_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
    }
    reference["reference_sha256"] = content_sha256(reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("vehicle_parameters.json", vehicle),
        ("geometry_manifest.json", manifest),
        ("reference_config.json", reference),
    ):
        serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
        (output_dir / name).write_text(serialized)
    return reference


def load_reference_bundle(
    path: Path, xml_path: Path | None = None,
) -> dict[str, Any]:
    """Load a frozen reference, checking artifacts and current source hashes.

    Args:
        path: reference_config.json beside vehicle/geometry JSON artifacts.
        xml_path: Runtime scene, if supplied; must match the frozen scene and
            its include closure. No runtime state or source files are edited.

    Returns:
        Validated self-contained reference configuration.

    Raises:
        ValueError: A fingerprint, schema or duplicated physical fact differs.
        FileNotFoundError: An artifact or referenced source is missing.
    """
    reference = json.loads(path.read_text())
    vehicle = json.loads((path.parent / "vehicle_parameters.json").read_text())
    geometry = json.loads((path.parent / "geometry_manifest.json").read_text())
    payload = {key: value for key, value in reference.items()
               if key != "reference_sha256"}
    if content_sha256(payload) != reference.get("reference_sha256"):
        raise ValueError("Reference fingerprint mismatch; re-export reference")
    if (content_sha256(vehicle) != reference.get("vehicle_sha256")
            or content_sha256(geometry) != reference.get("geometry_sha256")):
        raise ValueError("Vehicle/geometry fingerprint mismatch")
    if reference.get("kind") != "x4_950_component_drag_reference_v1":
        raise ValueError("Unsupported analytical reference schema")
    if reference.get("mass_total_kg") != vehicle.get("mass_total_kg"):
        raise ValueError("Reference mass disagrees with frozen vehicle")
    if reference.get("proxies") != geometry.get("proxies"):
        raise ValueError("Reference geometry disagrees with frozen manifest")
    if reference.get("initial_joints") != vehicle.get("initial_joints"):
        raise ValueError("Reference initial joints disagree with vehicle")
    dependencies = vehicle["dependencies"]
    if (content_sha256(dependencies) != reference["dependency_sha256"]
            or reference["dependency_sha256"] != vehicle["dependency_sha256"]):
        raise ValueError("Dependency fingerprint mismatch")
    if xml_path is not None:
        if _label(xml_path) != vehicle["scene"]:
            raise ValueError("Runtime XML scene differs from frozen reference")
        closure = {_label(item) for item in xml_dependency_closure(xml_path)}
        if closure != set(dependencies["xml_sha256"]):
            raise ValueError("Runtime XML include closure differs")
    for group in ("xml_sha256", "mesh_sha256"):
        for name, digest in dependencies[group].items():
            current = hashlib.sha256(
                (PROJECT_ROOT / name).read_bytes()
            ).hexdigest()
            if current != digest:
                raise ValueError(f"Dependency fingerprint mismatch: {name}")
    return reference
