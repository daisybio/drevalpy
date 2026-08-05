"""Functions for evaluating model performance."""

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn import metrics

from .datasets.dataset import DrugResponseDataset
from .utils._pipeline_function import pipeline_function

warning_shown = False
constant_prediction_warning_shown = False


def _check_constant_prediction(y_pred: np.ndarray) -> bool:
    """Check if predictions are constant.

    :param y_pred: Predicted values.

    :returns: Whether all predictions are equal within tolerance.
    """
    tol = 1e-6
    # no variation in predictions
    return bool(np.all(np.isclose(y_pred, y_pred[0], atol=tol)))


def _check_constant_target_or_small_sample(y_true: np.ndarray) -> bool:
    """Check if target is constant or sample size is too small.

    :param y_true: Observed response values.

    :returns: Whether the sample is too small or the target has no variation.
    """
    tol = 1e-6
    # Check for insufficient sample size or no variation in target
    return len(y_true) < 2 or bool(np.all(np.isclose(y_true, y_true[0], atol=tol)))


def pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Pearson correlation between predictions and response.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.

    :returns: Pearson correlation, or ``0.0`` / ``nan`` for degenerate inputs.

    :raises AssertionError: If ``y_pred`` and ``y_true`` differ in length.
    """
    if len(y_pred) != len(y_true):
        raise AssertionError("predictions, response  must have the same length")

    if _check_constant_prediction(y_pred):
        return 0.0
    if _check_constant_target_or_small_sample(y_true):
        return np.nan

    return pearsonr(y_pred, y_true)[0]


def spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Spearman correlation between predictions and response.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.

    :returns: Spearman correlation, or ``0.0`` / ``nan`` for degenerate inputs.

    :raises AssertionError: If ``y_pred`` and ``y_true`` differ in length.
    """
    # we can use scipy.stats.spearmanr
    if len(y_pred) != len(y_true):
        raise AssertionError("predictions, response  must have the same length")
    if _check_constant_prediction(y_pred):
        return 0.0
    if _check_constant_target_or_small_sample(y_true):
        return np.nan

    return spearmanr(y_pred, y_true)[0]


def kendall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Kendall tau correlation between predictions and response.

    :param y_pred: Predicted response values.
    :param y_true: Observed response values.

    :returns: Kendall tau, or ``0.0`` / ``nan`` for degenerate inputs.

    :raises AssertionError: If ``y_pred`` and ``y_true`` differ in length.
    """
    # we can use scipy.stats.spearmanr
    if len(y_pred) != len(y_true):
        raise AssertionError("predictions, response  must have the same length")
    if _check_constant_prediction(y_pred):
        return 0.0
    if _check_constant_target_or_small_sample(y_true):
        return np.nan

    return kendalltau(y_pred, y_true)[0]


AVAILABLE_METRICS = {
    "MSE": metrics.mean_squared_error,
    "RMSE": metrics.root_mean_squared_error,
    "MAE": metrics.mean_absolute_error,
    "R^2": metrics.r2_score,
    "Pearson": pearson,
    "Spearman": spearman,
    "Kendall": kendall,
}
# both used by pipeline!
MINIMIZATION_METRICS = ["MSE", "RMSE", "MAE"]
MAXIMIZATION_METRICS = ["R^2", "Pearson", "Spearman", "Kendall"]


def get_mode(metric: str):
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


def _should_return_nan_global(response: np.ndarray, predictions: np.ndarray) -> bool:
    return bool(len(response) < 2 or np.all(np.isnan(response)) or np.all(np.isnan(predictions)))


def _masked_metric_inputs(predictions: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.any(np.isnan(predictions)):
        return predictions, response
    if np.all(np.isnan(predictions)):
        return None
    mask = ~np.isnan(predictions)
    return predictions[mask], response[mask]


def _compute_metric_value(metric_name: str, predictions: np.ndarray, response: np.ndarray) -> float:
    if _should_return_nan_global(response, predictions):
        return float(np.nan)
    masked = _masked_metric_inputs(predictions, response)
    if masked is None:
        return float(np.nan)
    y_pred, y_true = masked
    return float(AVAILABLE_METRICS[metric_name](y_pred=y_pred, y_true=y_true))


@pipeline_function
def evaluate(dataset: DrugResponseDataset, metric: list[str] | str):
    """Compute evaluation metrics from stored predictions on a dataset.

    :param dataset: ``DrugResponseDataset`` with ``predictions`` populated.
    :param metric: One metric name or a list of names from ``AVAILABLE_METRICS``.

    :returns: Mapping from metric name to scalar score.

    :raises AssertionError: If predictions are missing or a metric name is unknown.
    """
    if isinstance(metric, str):
        metric = [metric]
    predictions = dataset.predictions
    if predictions is None:
        raise AssertionError("No predictions found in the dataset")
    response = dataset.response

    results = {}
    for m in metric:
        if m not in AVAILABLE_METRICS:
            raise AssertionError(f"invalid metric {m}. Available: {list(AVAILABLE_METRICS.keys())}")
        results[m] = _compute_metric_value(m, predictions, response)

    return results
