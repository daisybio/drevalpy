"""Tests for evaluation metric helpers."""

import numpy as np

from drevalpy.evaluation import _compute_metric_value


def test_compute_metric_value_all_nan_predictions() -> None:
    response = np.array([1.0, 2.0])
    predictions = np.array([np.nan, np.nan])
    assert np.isnan(_compute_metric_value("RMSE", predictions, response))
