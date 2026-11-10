"""Real small training/checkpoint and immutable dataset integration checks."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wind_field_simulation.training.dataset_v1 import generate_dataset, load_dataset
from wind_field_simulation.training.model_v1 import X4WindAeroV1, load_checkpoint, predict_increment
from wind_field_simulation.training.train_v1 import pair_loss, train_pilot
from wind_field_simulation.training.xml_parameters import (
    DEFAULT_SCENE,
    DEFAULT_ASSUMPTIONS,
    export_reference_bundle,
)


def test_dataset_train_checkpoint_roundtrip_and_corruption(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    export_reference_bundle(DEFAULT_SCENE, DEFAULT_ASSUMPTIONS, reference_dir)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"reference_config": str(reference_dir / "reference_config.json"),
                                  "split_counts": [2000, 500, 500], "sampling_seed": 7}))
    manifest = generate_dataset(config, tmp_path / "dataset")
    with pytest.raises(FileExistsError):
        generate_dataset(config, manifest.parent)
    data, metadata = load_dataset(manifest)
    assert metadata["split_counts"] == [2000, 500, 500]
    checkpoint = train_pilot(manifest, tmp_path / "training",
                             {"epochs": 12, "batch_size": 256})
    network, saved = load_checkpoint(checkpoint)
    assert saved["kind"] == "x4_950_wind_increment_xy_v1"
    assert saved["dataset_manifest_sha256"] == metadata["manifest_sha256"]
    history = json.loads((checkpoint.parent / "history.json").read_text())
    assert min(row["validation_loss"] for row in history[1:]) < history[0]["validation_loss"]
    train_inputs = np.concatenate((data["air_velocity_body_pair"],
                                  np.repeat(data["omega_body"][:, None, :], 2, axis=1)), axis=2)
    np.testing.assert_allclose(network.input_mean.numpy(),
                               train_inputs[data["split"] == 0].reshape(-1, 6).mean(axis=0),
                               atol=1e-7)
    zero = predict_increment(network, data["velocity_world"][:8],
                              data["rotation_body_to_world"][:8], data["omega_body"][:8],
                              np.zeros((8, 3)))
    np.testing.assert_array_equal(zero, np.zeros((8, 3)))
    with (manifest.parent / "pairs.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="fingerprint"):
        load_dataset(manifest)


def test_legacy_weights_rejected_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"state_dict": {"weight": torch.zeros((17, 64))}}, path)
    with pytest.raises(ValueError, match="kind"):
        load_checkpoint(path)


@pytest.mark.parametrize("acceleration,positive", [([-0.01, 0.02, 0.], True),
                                                    ([0.01, -0.02, 0.], False)])
def test_work_penalty_uses_physical_dot_product_with_unequal_scales(
    acceleration: list[float], positive: bool,
) -> None:
    scales = torch.tensor([.05, .5, .05])
    network = X4WindAeroV1(np.zeros(6), np.ones(6), scales.numpy(), np.zeros(3))
    with torch.no_grad():
        for parameter in network.residual.parameters():
            parameter.zero_()
        network.residual[-1].bias.copy_(torch.tensor(acceleration) / scales)
    air = torch.tensor([[[1., 1., 0.], [1., 1., 0.]]])
    batch = {"air": air, "omega": torch.zeros(1, 3),
             "rotation": torch.eye(3)[None], "full": torch.zeros(1, 2, 3),
             "delta": torch.zeros(1, 3)}
    base = pair_loss(network, batch, scales, scales, 0.)
    penalized = pair_loss(network, batch, scales, scales, 1.)
    if positive:
        assert penalized > base
    else:
        torch.testing.assert_close(penalized, base)
