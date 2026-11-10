"""Metrics must distinguish exact zero identities from small-label baselines."""

import numpy as np

from wind_field_simulation.training.evaluate_v1 import error_metrics


def test_metric_units_relative_denominator_and_zero_predictor() -> None:
    label = np.array([[3., 4., 0.], [0., 0., 0.]])
    prediction = np.array([[0., 0., 0.], [0., 0., 0.]])
    result = error_metrics(prediction, label, np.array([True, False]))
    np.testing.assert_allclose(result["axis_rmse_m_s2"], [3/np.sqrt(2), 4/np.sqrt(2)])
    assert result["normalized_nonzero_vector_rmse"] == 1
    assert result["vector_error_max_m_s2"] == 5
    zero = error_metrics(np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2, dtype=bool))
    assert zero["normalized_nonzero_vector_rmse"] is None
