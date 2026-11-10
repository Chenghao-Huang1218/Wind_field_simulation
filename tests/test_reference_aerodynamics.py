"""Independent hand calculations and physical identities for the drag teacher."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wind_field_simulation.training.reference_aerodynamics import (
    paired_wind_labels,
    reference_wrench_body,
)


def one_proxy(center: tuple[float, float, float] = (0, 0, 0)) -> dict[str, Any]:
    """Use rho=2, Cd=1 and A=[6,3,2] so expected forces are integers."""
    return {"mass_total_kg": 2.0,
            "assumptions": {"density_kg_m3": 2.0, "cd_axes": [1, 1, 1],
                            "wind_max_horizontal_m_s": 3.0},
            "proxies": [{"center_body_m": list(center),
                         "rotation_body_from_proxy": np.eye(3).tolist(),
                         "areas_m2": [6, 3, 2]}]}


def test_hand_calculated_axis_drag_and_quadratic_scaling() -> None:
    reference = one_proxy()
    force, torque = reference_wrench_body(np.array([2, -3, 0]), np.zeros(3), reference)
    np.testing.assert_array_equal(force, [-24, 27, 0])
    np.testing.assert_array_equal(torque, [0, 0, 0])
    twice, _ = reference_wrench_body(np.array([4, -6, 0]), np.zeros(3), reference)
    np.testing.assert_array_equal(twice, 4 * force)
    zero, _ = reference_wrench_body(np.zeros(3), np.zeros(3), reference)
    np.testing.assert_array_equal(zero, [0, 0, 0])


def test_point_velocity_and_moment_are_about_root_origin() -> None:
    # omega x [2,0,0] = [0,2,0]; f=[0,-12,0]; r x f=[0,0,-24].
    force, torque = reference_wrench_body(np.zeros(3), np.array([0, 0, 1]),
                                         one_proxy((2, 0, 0)))
    np.testing.assert_array_equal(force, [0, -12, 0])
    np.testing.assert_array_equal(torque, [0, 0, -24])


def test_proxy_rotation_is_applied_before_axis_drag() -> None:
    reference = one_proxy()
    reference["proxies"][0]["rotation_body_from_proxy"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    force, _ = reference_wrench_body(np.array([2, 0, 0]), np.zeros(3), reference)
    np.testing.assert_array_equal(force, [-12, 0, 0])


def test_multiple_proxies_sum_and_full_power_is_dissipative() -> None:
    reference = one_proxy((2, 0, 0))
    reference["proxies"].append(one_proxy((-2, 0, 0))["proxies"][0])
    force, torque = reference_wrench_body(np.zeros(3), np.array([0, 0, 1]), reference)
    np.testing.assert_array_equal(force, [0, 0, 0])
    np.testing.assert_array_equal(torque, [0, 0, -48])
    rng = np.random.default_rng(42)
    velocity = rng.uniform(-6, 6, (256, 3))
    omega = rng.uniform(-1, 1, (256, 3))
    force, torque = reference_wrench_body(velocity, omega, reference)
    power = np.sum(force * velocity + torque * omega, axis=-1)
    assert np.all(power <= 1e-12)
    np.testing.assert_allclose(reference_wrench_body(-velocity, -omega, reference)[0], -force)


def test_same_state_pair_and_horizontal_projection() -> None:
    reference = one_proxy()
    velocity, quaternion = np.array([1, 0, 2]), np.array([1, 0, 0, 0])
    omega, wind = np.zeros(3), np.array([2, 0, 0])
    before = [value.copy() for value in (velocity, quaternion, omega, wind)]
    result = paired_wind_labels(velocity, quaternion, omega, wind, reference)
    np.testing.assert_array_equal(result["force_wind_world_n"], [6, 0, -8])
    np.testing.assert_array_equal(result["force_no_wind_world_n"], [-6, 0, -8])
    np.testing.assert_array_equal(result["wind_increment_acceleration_world_m_s2"], [6, 0, 0])
    for actual, original in zip((velocity, quaternion, omega, wind), before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_body_world_wxyz_transform_and_yaw_covariance() -> None:
    reference = one_proxy()
    half = np.sqrt(0.5)
    result = paired_wind_labels(np.zeros(3), np.array([half, 0, 0, half]),
                                np.zeros(3), np.array([0, 2, 0]), reference)
    np.testing.assert_allclose(result["force_wind_world_n"], [0, 24, 0], atol=1e-13)
    np.testing.assert_allclose(result["wind_increment_acceleration_world_m_s2"], [0, 12, 0], atol=1e-13)
    # 90 deg pitch rotates root +X into world -Z; preserve full Z but project label.
    tilted = paired_wind_labels(np.array([0, 0, -2]), np.array([half, 0, half, 0]),
                                np.zeros(3), np.zeros(3), reference)
    np.testing.assert_allclose(tilted["force_no_wind_world_n"], [0, 0, 24], atol=1e-13)


def test_batch_broadcast_and_exact_zero_wind() -> None:
    rng = np.random.default_rng(7)
    velocity = rng.normal(size=(12, 3))
    omega = rng.normal(size=(12, 3))
    result = paired_wind_labels(velocity, np.array([1, 0, 0, 0]), omega,
                                np.zeros(3), one_proxy((1, 0, 0)))
    np.testing.assert_array_equal(result["wind_increment_acceleration_world_m_s2"], np.zeros((12, 3)))
    wind = np.array([1, -2, 0])
    batched = paired_wind_labels(velocity, np.array([1, 0, 0, 0]), omega, wind, one_proxy())
    for index in range(12):
        scalar = paired_wind_labels(velocity[index], np.array([1, 0, 0, 0]), omega[index], wind, one_proxy())
        for key in scalar:
            np.testing.assert_allclose(batched[key][index], scalar[key], atol=1e-13)


@pytest.mark.parametrize("wind", [[3, 3, 0], [0, 0, 0.001], [3.000001, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]])
def test_rejects_out_of_domain_wind(wind: list[float]) -> None:
    with pytest.raises(ValueError):
        paired_wind_labels(np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3),
                           np.asarray(wind), one_proxy())


def test_opposing_velocity_allows_six_m_s_relative_air_speed() -> None:
    result = paired_wind_labels(np.array([-3, 0, 0]), np.array([1, 0, 0, 0]),
                                np.zeros(3), np.array([3, 0, 0]), one_proxy())
    np.testing.assert_array_equal(result["force_wind_world_n"], [216, 0, 0])
    np.testing.assert_array_equal(result["wind_increment_acceleration_world_m_s2"], [81, 0, 0])


@pytest.mark.parametrize("quaternion", [[0, 0, 0, 0], [2, 0, 0, 0], [np.nan, 0, 0, 1]])
def test_rejects_invalid_quaternion(quaternion: list[float]) -> None:
    with pytest.raises(ValueError):
        paired_wind_labels(np.zeros(3), np.asarray(quaternion), np.zeros(3),
                           np.zeros(3), one_proxy())


@pytest.mark.parametrize("field", ["density", "cd", "area", "rotation", "mass"])
def test_rejects_nonphysical_reference(field: str) -> None:
    reference = copy.deepcopy(one_proxy())
    if field == "density":
        reference["assumptions"]["density_kg_m3"] = -1
    elif field == "cd":
        reference["assumptions"]["cd_axes"][0] = -1
    elif field == "area":
        reference["proxies"][0]["areas_m2"][0] = 0
    elif field == "rotation":
        reference["proxies"][0]["rotation_body_from_proxy"][0][0] = -1
    else:
        reference["mass_total_kg"] = 0
    with pytest.raises(ValueError):
        paired_wind_labels(np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3),
                           np.zeros(3), reference)


def test_rejects_nonfinite_state_or_wrong_shape() -> None:
    for velocity in (np.array([np.nan, 0, 0]), np.zeros(2)):
        with pytest.raises(ValueError):
            reference_wrench_body(velocity, np.zeros(3), one_proxy())


def test_inspection_cli_checks_real_reference_and_emits_metrics(tmp_path: Path) -> None:
    from wind_field_simulation.training.xml_parameters import (
        DEFAULT_ASSUMPTIONS, DEFAULT_SCENE, export_reference_bundle,
    )

    export_reference_bundle(DEFAULT_SCENE, DEFAULT_ASSUMPTIONS, tmp_path)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "scripts/check_x4_950_reference.py"),
        "--reference", str(tmp_path / "reference_config.json"),
        "--samples", "32", "--timing-repeats", "10",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "reference_check.json").read_text())
    assert report["seed"] == 42
    assert report["state_samples"] == 32
    assert report["max_full_drag_power_w"] <= 1e-10
    assert report["zero_wind_max_abs_increment_m_s2"] == 0
    assert report["label_z_max_abs_m_s2"] == 0
    assert report["scalar_pair_timing_us"]["median"] > 0
    assert len(report["rest_wind_grid"]) == 169  # 7 x 24 + opposing-wind case.
    # A modified frozen artifact must be rejected before checking or timing it.
    config_path = tmp_path / "reference_config.json"
    reference = json.loads(config_path.read_text())
    reference["mass_total_kg"] = 6.5
    config_path.write_text(json.dumps(reference))
    rejected = subprocess.run([
        sys.executable, str(root / "scripts/check_x4_950_reference.py"),
        "--reference", str(config_path), "--timing-repeats", "1",
    ], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "fingerprint" in rejected.stderr
