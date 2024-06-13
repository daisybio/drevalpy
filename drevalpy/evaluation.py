from .datasets.dataset import DrugResponseDataset
from typing import Union, List
import sklearn.metrics as metrics
from .utils import pearson, spearman, kendall, partial_correlation
import pandas as pd
import numpy as np

AVAILABLE_METRICS = {
    "MSE": metrics.mean_squared_error,
    "RMSE": metrics.root_mean_squared_error,
    "MAE": metrics.mean_absolute_error,
    "R^2": metrics.r2_score,
    "Pearson": pearson,
    "Spearman": spearman,
    "Kendall": kendall,
    "Partial_Correlation": partial_correlation,
}
MINIMIZATION_METRICS = ["MSE", "RMSE", "MAE"]
MAXIMIZATION_METRICS = ["R^2", "Pearson", "Spearman", "Kendall", "Partial_Correlation"]


def get_mode(metric: str):
    if metric in MINIMIZATION_METRICS:
        mode = "min"
    elif metric in MAXIMIZATION_METRICS:
        mode = "max"
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Need to add metric to MINIMIZATION_METRICS or MAXIMIZATION_METRICS?"
        )
    return mode


def evaluate(dataset: DrugResponseDataset, metric: Union[List[str], str]):
    """
    Evaluates the model on the given dataset.
    :param dataset: dataset to evaluate on
    :param metric: evaluation metric(s) (one or a list of "mse", "rmse", "mae", "r2", "pearson", "spearman", "kendall", "partial_correlation")
    :return: evaluation metric
    """
    if isinstance(metric, str):
        metric = [metric]
    predictions = dataset.predictions
    response = dataset.response
    results = {}
    for m in metric:
        assert (
            m in AVAILABLE_METRICS
        ), f"invalid metric {m}. Available: {list(AVAILABLE_METRICS.keys())}"
        if len(response) < 2:
            results[m] = float(np.nan)
        else:
            if m == "Partial_Correlation":
                results[m] = float(
                    AVAILABLE_METRICS[m](
                        y_pred=predictions,
                        y_true=response,
                        cell_line_ids=dataset.cell_line_ids,
                        drug_ids=dataset.drug_ids,
                    )
                )
            else:
                results[m] = float(
                    AVAILABLE_METRICS[m](y_pred=predictions, y_true=response)
                )

    return results


def visualize_results(results: pd.DataFrame, mode: Union[List[str], str]):
    """
    Visualizes the model on the given dataset.
    :param dataset: dataset to evaluate on
    :mode
    :return: evaluation metric
    """
    raise NotImplementedError("visualize not implemented yet")
