"""Functions for evaluating model performance.

``scipy.stats`` and ``sklearn.metrics`` are imported inside the metric functions
rather than at module scope. This module is on the critical path of
``import drevalpy`` - :mod:`drevalpy.types.results.experiment` needs
:data:`AVAILABLE_METRICS` - and those two libraries cost ~0.3s between them,
which every CLI invocation would otherwise pay before computing a single metric.
See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


#: Absolute tolerance of the constant check; ``np.isclose`` adds its default relative tolerance (``1e-5``).
CONSTANT_ATOL = 1e-6


def ranges_are_constant(high: np.ndarray | float, low: np.ndarray | float) -> np.ndarray:
    """Decide elementwise whether a series with maximum ``high`` and minimum ``low`` is constant.

    This is the one rule behind :func:`is_constant` and the grouped correlations in
    :mod:`drevalpy._grouped_pearson`: ``np.isclose(high, low, atol=CONSTANT_ATOL)``, so the
    default relative tolerance applies and a series offset by ``1e5`` may span about ``1``
    and still count as constant. Because it only looks at the maximum and minimum, the
    verdict does not depend on the order of the values.

    :param high: Maximum of each series.
    :param low: Minimum of each series.

    :returns: Boolean array (or scalar) of the same shape as the inputs.
    """
    return np.isclose(high, low, atol=CONSTANT_ATOL)


def is_constant(values: np.ndarray) -> bool:
    """Check whether all values are equal within tolerance, independently of their order.

    :param values: Non-empty array of values.

    :returns: Whether the maximum and minimum of ``values`` are close, see :func:`ranges_are_constant`.
    """
    return bool(ranges_are_constant(np.max(values), np.min(values)))


def _check_constant_prediction(y_pred: np.ndarray) -> bool:
    """Check if predictions are constant.

    :param y_pred: Predicted values.

    :returns: Whether all predictions are equal within tolerance.
    """
    return is_constant(y_pred)


def _check_constant_target_or_few_pairs(y_true: np.ndarray) -> bool:
    """Check if target is constant or there are too few pairs.

    :param y_true: Observed response values.

    :returns: Whether there are too few pairs or the target has no variation.
    """
    return len(y_true) < 2 or is_constant(y_true)


def pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Pearson correlation between predictions and response.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.

    :returns: Pearson correlation, or ``0.0`` / ``nan`` for degenerate inputs.

    :raises ValueError: If ``y_pred`` and ``y_true`` differ in length.
    """
    if len(y_pred) != len(y_true):
        raise ValueError("predictions, response must have the same length")

    if _check_constant_prediction(y_pred):
        return 0.0
    if _check_constant_target_or_few_pairs(y_true):
        return np.nan

    from scipy.stats import pearsonr

    return pearsonr(y_pred, y_true)[0]


def spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Spearman correlation between predictions and response.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.

    :returns: Spearman correlation, or ``0.0`` / ``nan`` for degenerate inputs.

    :raises ValueError: If ``y_pred`` and ``y_true`` differ in length.
    """
    # we can use scipy.stats.spearmanr
    if len(y_pred) != len(y_true):
        raise ValueError("predictions, response must have the same length")
    if _check_constant_prediction(y_pred):
        return 0.0
    if _check_constant_target_or_few_pairs(y_true):
        return np.nan

    from scipy.stats import spearmanr

    return spearmanr(y_pred, y_true)[0]


def kendall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Kendall tau correlation between predictions and response.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.

    :returns: Kendall tau, or ``0.0`` / ``nan`` for degenerate inputs.

    :raises ValueError: If ``y_pred`` and ``y_true`` differ in length.
    """
    # we can use scipy.stats.spearmanr
    if len(y_pred) != len(y_true):
        raise ValueError("predictions, response must have the same length")
    if _check_constant_prediction(y_pred):
        return 0.0
    if _check_constant_target_or_few_pairs(y_true):
        return np.nan

    from scipy.stats import kendalltau

    return kendalltau(y_pred, y_true)[0]


def _mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error, deferring the ``sklearn`` import to call time.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.
    :returns: Mean squared error.
    """
    from sklearn.metrics import mean_squared_error

    return float(mean_squared_error(y_true=y_true, y_pred=y_pred))


def _root_mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root mean squared error, deferring the ``sklearn`` import to call time.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.
    :returns: Root mean squared error.
    """
    from sklearn.metrics import root_mean_squared_error

    return float(root_mean_squared_error(y_true=y_true, y_pred=y_pred))


def _mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean absolute error, deferring the ``sklearn`` import to call time.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.
    :returns: Mean absolute error.
    """
    from sklearn.metrics import mean_absolute_error

    return float(mean_absolute_error(y_true=y_true, y_pred=y_pred))


def _r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination, deferring the ``sklearn`` import to call time.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.
    :returns: R^2 score.
    """
    from sklearn.metrics import r2_score

    return float(r2_score(y_true=y_true, y_pred=y_pred))


AVAILABLE_METRICS = {
    "MSE": _mean_squared_error,
    "RMSE": _root_mean_squared_error,
    "MAE": _mean_absolute_error,
    "R^2": _r2_score,
    "Pearson": pearson,
    "Spearman": spearman,
    "Kendall": kendall,
}
# both used by pipeline!
MINIMIZATION_METRICS = ["MSE", "RMSE", "MAE"]
MAXIMIZATION_METRICS = ["R^2", "Pearson", "Spearman", "Kendall"]


def get_mode(metric: str) -> Literal["min", "max"]:
    """Return whether lower or higher metric values are better.

    :param metric: Metric name (for example ``"RMSE"`` or ``"Pearson"``).

    :returns: ``"min"`` for error metrics or ``"max"`` for correlation metrics.

    :raises ValueError: If ``metric`` is not a known minimization or maximization metric.
    """
    if metric in MINIMIZATION_METRICS:
        mode = "min"
    elif metric in MAXIMIZATION_METRICS:
        mode = "max"
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Need to add metric to MINIMIZATION_METRICS or MAXIMIZATION_METRICS?"
        )
    return mode


def score_predictions(
    predictions: np.ndarray,
    response: np.ndarray,
    metrics: list[str] | None = None,
    *,
    empty_as_dict: bool = False,
) -> dict[str, float]:
    """Evaluate aligned predictions with one shared missing-response mask.

    :param predictions: One prediction per requested response position.
    :param response: Targets in exactly the same order, including NaNs.
    :param metrics: Requested metric names; None selects every registered metric.
    :param empty_as_dict: Return {} when no pair is usable (experiment convention).
    :returns: Requested scores, with NaN for too few pairs.
    :raises ValueError: If inputs are not equally shaped one-dimensional arrays or a metric is unknown.
    """
    predictions, response = np.asarray(predictions), np.asarray(response)
    if predictions.ndim != 1 or response.ndim != 1 or predictions.shape != response.shape:
        raise ValueError("Predictions and targets must be equal-shaped one-dimensional arrays")
    names = list(AVAILABLE_METRICS) if metrics is None else metrics
    for name in names:
        if name not in AVAILABLE_METRICS:
            raise ValueError(f"invalid metric {name}. Available: {list(AVAILABLE_METRICS)}")
    valid = ~np.isnan(predictions) & ~np.isnan(response)
    if empty_as_dict and not valid.any():
        return {}
    y_pred, y_true = predictions[valid], response[valid]
    return {
        name: (
            float(AVAILABLE_METRICS[name](y_pred=y_pred, y_true=y_true))
            if len(y_true) >= (1 if name in MINIMIZATION_METRICS else 2)
            else float("nan")
        )
        for name in names
    }


def evaluate(
    predictions_or_dataset: Any = None,
    response: np.ndarray | pd.Series | None = None,
    metric: list[str] | str = "Pearson",
    *,
    predictions: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics from predictions and observed response.

    Accepts either (predictions_array, response_array) or a single
    object that carries both .predictions and .response.

    :param predictions_or_dataset: Predicted values or a dataset object.
    :param response: Observed (true) response values (omit if dataset passed).
    :param metric: One metric name or a list of names from ``AVAILABLE_METRICS``.
    :param predictions: Keyword-only alias for predictions_or_dataset.
    :returns: Mapping from metric name to scalar score.
    :raises ValueError: If predictions or response are missing, the inputs are not equal-shaped
        one-dimensional arrays, or a metric name is unknown.
    """
    if isinstance(metric, str):
        metric = [metric]

    if predictions is not None:
        preds_arr, response_arr = predictions, response
    elif (
        response is None
        and hasattr(predictions_or_dataset, "predictions")
        and hasattr(predictions_or_dataset, "response")
    ):
        preds_arr = predictions_or_dataset.predictions
        response_arr = predictions_or_dataset.response
    else:
        preds_arr, response_arr = predictions_or_dataset, response
    if preds_arr is None or response_arr is None:
        raise ValueError("evaluate needs predictions and response, or one object carrying both")

    return score_predictions(np.asarray(preds_arr), np.asarray(response_arr), metric)
