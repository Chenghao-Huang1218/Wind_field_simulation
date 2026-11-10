"""Pilot sampling must be paired, reproducible, bounded and leakage free."""

import numpy as np
import pytest
from pathlib import Path

from wind_field_simulation.training.dataset_v1 import (
    build_pilot_dataset,
    check_split_leakage,
)
from wind_field_simulation.training.xml_parameters import (
    DEFAULT_SCENE,
    DEFAULT_ASSUMPTIONS,
    export_reference_bundle,
    load_reference_bundle,
)


def test_pilot_reproducibility_bounds_pair_and_groups(tmp_path: Path) -> None:
    export_reference_bundle(DEFAULT_SCENE, DEFAULT_ASSUMPTIONS, tmp_path)
    reference = load_reference_bundle(
        tmp_path / "reference_config.json"
    )
    data = build_pilot_dataset(reference, seed=7, split_counts=(40, 20, 20))
    again = build_pilot_dataset(reference, seed=7, split_counts=(40, 20, 20))
    for name in data:
        np.testing.assert_array_equal(data[name], again[name])
        assert np.isfinite(data[name]).all()
    assert np.bincount(data["split"]).tolist() == [40, 20, 20]
    assert np.max(np.linalg.norm(data["wind_world"][:, :2], axis=1)) <= 3 + 1e-14
    assert not data["wind_world"][:, 2].any()
    assert np.max(np.linalg.norm(data["velocity_world"][:, :2], axis=1)) <= 3 + 1e-14
    assert np.max(np.abs(data["velocity_world"][:, 2])) <= 1
    assert np.max(np.linalg.norm(data["omega_body"], axis=1)) <= 1.5
    assert np.max(np.abs(data["euler_rpy_rad"][:, :2])) <= np.deg2rad(20)
    assert not data["delta_acceleration_world"][:, 2].any()
    zero = np.all(data["wind_world"] == 0, axis=1)
    assert zero.any()
    np.testing.assert_array_equal(data["delta_acceleration_world"][zero], 0)
    np.testing.assert_allclose(
        np.einsum("nji,nkj->nki", data["rotation_body_to_world"],
                  np.stack((data["force_wind_world_n"],
                            data["force_no_wind_world_n"]), axis=1))
        / reference["mass_total_kg"], data["full_acceleration_body"], atol=1e-15,
    )
    assert check_split_leakage(data)["cross_split_core_duplicates"] == 0
    assert len(set(data["group_id"])) == 8
    for group in np.unique(data["group_id"]):
        assert len(np.unique(data["split"][data["group_id"] == group])) == 1


def test_transformed_core_leakage_cannot_hide_in_different_world_states() -> None:
    data = {"split": np.array([0, 1]), "group_id": np.array([0, 1]),
            "air_velocity_body_pair": np.ones((2, 2, 3)),
            "omega_body": np.zeros((2, 3))}
    with pytest.raises(ValueError, match="core"):
        check_split_leakage(data)
