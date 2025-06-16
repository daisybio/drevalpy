"""Functions for evaluating model performance."""

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn import metrics

from .datasets.dataset import DrugResponseDataset
from .pipeline_function import pipeline_function

warning_shown = False
constant_prediction_warning_shown = False


def _check_constant_prediction(y_pred: np.ndarray) -> bool:
    """
    Check if predictions are constant.

    :param y_pred: predictions
    :return: bool whether predictions are constant
    """
    tol = 1e-6
    # no variation in predictions
    return bool(np.all(np.isclose(y_pred, y_pred[0], atol=tol)))


def _check_constant_target_or_small_sample(y_true: np.ndarray) -> bool:
    """
    Check if target is constant or sample size is too small.

    :param y_true: true response
    :returns: bool whether target is constant or sample size is too small
    """
    tol = 1e-6
    # Check for insufficient sample size or no variation in target
    return len(y_true) < 2 or bool(np.all(np.isclose(y_true, y_true[0], atol=tol)))


def pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the pearson correlation between predictions and response.

    :param y_pred: predictions
    :param y_true: response
    :return: pearson correlation float
    :raises AssertionError: if predictions and response do not have the same length
    """
    if len(y_pred) != len(y_true):
        raise AssertionError("predictions, response  must have the same length")

    if _check_constant_prediction(y_pred):
        return 0.0
    if _check_constant_target_or_small_sample(y_true):
        return np.nan

    return pearsonr(y_pred, y_true)[0]


def spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the spearman correlation between predictions and response.

    :param y_pred: predictions
    :param y_true: response
    :return: spearman correlation float
    :raises AssertionError: if predictions and response do not have the same length
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
    """
    Computes the kendall tau correlation between predictions and response.

    :param y_pred: predictions
    :param y_true: response
    :return: kendall tau correlation float
    :raises AssertionError: if predictions and response do not have the same length
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
    """
    Get whether the optimum value of the metric is the minimum or maximum.

    :param metric: metric, e.g., RMSE
    :returns: whether the optimum value of the metric is the minimum or maximum
    :raises ValueError: if the metric is not in MINIMIZATION_METRICS or MAXIMIZATION_METRICS
    """
    if metric in MINIMIZATION_METRICS:
        mode = "min"
    elif metric in MAXIMIZATION_METRICS:
        mode = "max"
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Need to add metric to MINIMIZATION_METRICS or " f"MAXIMIZATION_METRICS?"
        )
    return mode


@pipeline_function
def evaluate(dataset: DrugResponseDataset, metric: list[str] | str):
    """
    Evaluates the model on the given dataset.

    :param dataset: dataset to evaluate on
    :param metric: evaluation metric(s) (one or a list of "MSE", "RMSE", "MAE", "R^2", "Pearson",
        "spearman", "kendall")
    :return: evaluation metric
    :raises AssertionError: if metric is not in AVAILABLE
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
        if len(response) < 2 or np.all(np.isnan(response)) or np.all(np.isnan(predictions)):
            results[m] = float(np.nan)
        else:
            # check whether the predictions contain NaNs
            if np.any(np.isnan(predictions)):
                # if there are only NaNs in the predictions, the metric is NaN
                if np.all(np.isnan(predictions)):
                    results[m] = float(np.nan)
                else:
                    # remove the rows with NaNs in the predictions and response
                    mask = ~np.isnan(predictions)
                    results[m] = float(AVAILABLE_METRICS[m](y_pred=predictions[mask], y_true=response[mask]))
            else:
                results[m] = float(AVAILABLE_METRICS[m](y_pred=predictions, y_true=response))

    return results
