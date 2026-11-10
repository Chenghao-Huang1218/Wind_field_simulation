"""Numerical contracts for the separate X4 pilot surrogate."""

import numpy as np
import pytest
import torch

from wind_field_simulation.training.model_v1 import (
    X4WindAeroV1,
    fit_normalization,
    predict_increment,
)


def model() -> X4WindAeroV1:
    torch.manual_seed(42)
    return X4WindAeroV1(np.zeros(6), np.ones(6), np.ones(3) * 0.05,
                        np.array([0.01, 0.02, 0.03]))


def test_fixed_prior_output_shape_and_normalization_training_only() -> None:
    network = model()
    velocity = torch.tensor([[2., -3., 1.]])
    with torch.no_grad():
        for parameter in network.residual.parameters():
            parameter.zero_()
    torch.testing.assert_close(network(velocity, torch.zeros_like(velocity)),
                               torch.tensor([[-0.04, 0.18, -0.03]]))
    with pytest.raises(ValueError, match="N,3"):
        network(torch.zeros(1, 17), torch.zeros(1, 3))
    inputs = np.arange(36, dtype=float).reshape(3, 2, 6)
    full = np.arange(18, dtype=float).reshape(3, 2, 3) / 1000
    delta = np.zeros((3, 3))
    stats = fit_normalization(inputs, full, delta, np.array([True, False, False]))
    np.testing.assert_allclose(stats["input_mean"], inputs[0].mean(axis=0))
    assert np.min(stats["full_scale"]) >= 0.05
    assert np.min(stats["delta_scale"]) >= 0.05


def test_pair_zero_wind_exact_and_frame_rotation_covariance() -> None:
    network = model().eval()
    velocity = np.array([[1., 2., -0.5], [-1., 1., 0.]])
    omega = np.zeros((2, 3))
    rotation = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    wind = np.array([[1., 0., 0.], [0., 1., 0.]])
    zero = predict_increment(network, velocity, rotation, omega, np.zeros((2, 3)))
    np.testing.assert_array_equal(zero, np.zeros((2, 3)))
    original = predict_increment(network, velocity, rotation, omega, wind)
    world_yaw = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated = predict_increment(network, velocity @ world_yaw.T,
                                world_yaw @ rotation, omega, wind @ world_yaw.T)
    np.testing.assert_allclose(rotated, original @ world_yaw.T, atol=1e-8)
    assert not original[:, 2].any()
    with pytest.raises(ValueError, match="wind"):
        predict_increment(network, velocity, rotation, omega, np.ones((2, 3)))
