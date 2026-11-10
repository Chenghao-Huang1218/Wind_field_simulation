"""Guard XML-derived facts, geometry deduplication and source provenance."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from wind_field_simulation.training.xml_parameters import (
    export_vehicle_parameters,
    xml_dependency_closure,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "src/wind_field_simulation/assets/scenes/default/robot_quadrotor_950_with_arm.xml"


@pytest.fixture(scope="module")
def vehicle() -> dict[str, Any]:
    """Export the real default scene without stepping its dynamics."""
    return export_vehicle_parameters(SCENE)


def test_mass_com_inertia_and_frozen_joints(vehicle: dict[str, Any]) -> None:
    """Catch base-mass substitution, root/CoM confusion and pose drift."""
    assert vehicle["mass_total_kg"] == pytest.approx(8.0004)
    assert vehicle["base_body_mass_kg"] == 6.5
    np.testing.assert_allclose(vehicle["com_body_m"], [0, 0, -0.05872472], atol=5e-8)
    audit = json.loads((ROOT / "docs/provenance/x4-950-xml-audit-2026-11-10.json").read_text())
    np.testing.assert_allclose(vehicle["inertia_com_body_kg_m2"],
                               audit["assembly_inertia_about_com_body_kg_m2"], atol=1e-12)
    assert vehicle["initial_joints"]["joint1"]["qpos"] == [0.0]
    assert vehicle["initial_joints"]["endleft"]["qpos"] == [0.0]
    assert vehicle["initial_joints"]["endright"]["qpos"] == [0.0]
    assert len(vehicle["bodies"]) == 17
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert sum(body["mass_kg"] for body in vehicle["bodies"]) == pytest.approx(
        model.body_subtreemass[model.body("quadrotor_950_pos").id])


def test_rotor_binding_and_separate_thrust_limits(vehicle: dict[str, Any]) -> None:
    expected_xy = [(1, -1), (-1, 1), (1, 1), (-1, -1)]
    for rotor, (sign_x, sign_y) in zip(vehicle["rotors"], expected_xy, strict=True):
        np.testing.assert_allclose(rotor["position_body_m"],
                                   [sign_x * 0.3358757, sign_y * 0.3358757, 0.065])
        assert rotor["ctrlrange"] == [0, 61.2915625]
    assert vehicle["bridge_thrust_mapping"]["full_scale_per_motor_n"] == pytest.approx(39.241962)


def test_geometry_has_one_owner_and_no_rotor_or_scene_duplicates(vehicle: dict[str, Any]) -> None:
    manifest = vehicle["geometry_manifest"]
    proxies = manifest["proxies"]
    assert len(proxies) == 29  # 20 aircraft collision pieces + 9 arm bodies.
    selected_ids = [source["geom_id"] for item in proxies for source in item["sources"]]
    excluded_ids = [item["geom_id"] for item in manifest["excluded_geoms"]]
    assert len(selected_ids) == len(set(selected_ids))
    assert set(selected_ids).isdisjoint(excluded_ids)
    assert sorted(selected_ids + excluded_ids) == list(range(53))
    for proxy in proxies:
        assert not proxy["id"].startswith(("thruster", "rotor"))
        assert proxy["mass_owner"] != "world"
        radii = np.asarray(proxy["semiaxes_m"])
        assert np.all(radii > 0)
        np.testing.assert_allclose(proxy["areas_m2"], np.pi * np.prod(radii) / radii)
        rotation = np.asarray(proxy["rotation_body_from_proxy"])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    arm = {item["id"]: item for item in proxies if item["mass_owner"] != "quadrotor_950"}
    assert arm["link5"]["method"] == "visual_mesh_body_aabb"
    assert len(arm["link5"]["sources"]) == 2
    assert arm["link2"]["method"] == "collision_union_body_aabb"
    np.testing.assert_allclose(proxies[0]["semiaxes_m"], [0.13, 0.13, 0.08])
    capsule = next(item for item in proxies if item["id"] == "collision-arm-a00")
    np.testing.assert_allclose(capsule["semiaxes_m"], [0.01, 0.01, 0.24758787847868])


def test_mesh_bounds_are_in_body_frame_not_compiler_principal_frame(vehicle: dict[str, Any]) -> None:
    """Catch forgetting MuJoCo's mesh centering/rotation when deriving bounds."""
    obj = ROOT / "src/wind_field_simulation/assets/robot_models/airbot_play/meshes/link1.obj"
    vertices = np.array([[float(value) for value in line.split()[1:4]]
                         for line in obj.read_text().splitlines() if line.startswith("v ")])
    proxy = next(item for item in vehicle["geometry_manifest"]["proxies"] if item["id"] == "link1")
    np.testing.assert_allclose(proxy["semiaxes_m"], np.ptp(vertices, axis=0) / 2, atol=2e-8)


def test_actual_xml_and_referenced_mesh_hashes(vehicle: dict[str, Any]) -> None:
    dependencies = vehicle["dependencies"]
    xmls = dependencies["xml_sha256"]
    meshes = dependencies["mesh_sha256"]
    assert len(xmls) == 11
    assert len(meshes) == 18
    assert not any("cabinet" in path or "camera_stand" in path for path in [*xmls, *meshes])
    for path, digest in {**xmls, **meshes}.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert export_vehicle_parameters(SCENE) == vehicle


def test_recursive_include_closure(tmp_path: Path) -> None:
    """All nested XML changes must affect provenance, even without new geoms."""
    (tmp_path / "main.xml").write_text('<mujoco><include file="part.xml"/></mujoco>')
    (tmp_path / "part.xml").write_text('<mujocoinclude><include file="leaf.xml"/></mujocoinclude>')
    (tmp_path / "leaf.xml").write_text('<mujocoinclude/>')
    assert set(xml_dependency_closure(tmp_path / "main.xml")) == {
        tmp_path / name for name in ("main.xml", "part.xml", "leaf.xml")}


def test_nested_includes_use_main_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "main.xml").write_text('<mujoco><include file="sub/part.xml"/></mujoco>')
    (tmp_path / "sub/part.xml").write_text('<mujocoinclude><include file="leaf.xml"/></mujocoinclude>')
    (tmp_path / "leaf.xml").write_text('<mujocoinclude/>')
    assert set(xml_dependency_closure(tmp_path / "main.xml")) == {
        tmp_path / "main.xml", tmp_path / "sub/part.xml", tmp_path / "leaf.xml"}


def test_export_cli_writes_self_contained_reference(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts/export_x4_950_parameters.py"),
                             "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    reference = json.loads((tmp_path / "reference_config.json").read_text())
    assert reference["mass_total_kg"] == pytest.approx(8.0004)
    assert len(reference["proxies"]) == 29
    assert reference["assumptions"]["density_kg_m3"] == 1.225
    assert reference["assumptions"]["cd_axes"] == [1, 1, 1]
    assert (tmp_path / "vehicle_parameters.json").is_file()
    assert (tmp_path / "geometry_manifest.json").is_file()
