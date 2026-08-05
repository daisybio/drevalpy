"""Normalize evaluation metrics using NaiveMeanEffectsPredictor baselines."""

from __future__ import annotations

import pandas as pd

from ..datasets.dataset import DrugResponseDataset
from ..evaluation import AVAILABLE_METRICS, evaluate


def _index_naive_predictions(true_vs_pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    naive_mean_effects_dict: dict[str, pd.DataFrame] = {}
    for rand_setting in true_vs_pred["rand_setting"].unique():
        for test_mode in true_vs_pred["test_mode"].unique():
            key = f"{test_mode}_{rand_setting}"
            naive_mean_effects_dict[key] = true_vs_pred[
                (true_vs_pred["algorithm"] == "NaiveMeanEffectsPredictor")
                & (true_vs_pred["rand_setting"] == rand_setting)
                & (true_vs_pred["test_mode"] == test_mode)
            ]
    return naive_mean_effects_dict


def _adjust_setting_subset(
    setting_subset: pd.DataFrame,
    naive_mean_effects: pd.DataFrame,
) -> pd.DataFrame:
    naive_subset = naive_mean_effects[["drug_name", "cell_line_name", "CV_split", "y_pred"]].rename(
        columns={"y_pred": "y_pred_naive"}
    )
    merged = setting_subset[["drug_name", "cell_line_name", "CV_split", "y_true", "y_pred"]].merge(
        naive_subset, on=["drug_name", "cell_line_name", "CV_split"], how="left"
    )
    merged["y_true"] = merged["y_true"] - merged["y_pred_naive"]
    merged["y_pred"] = merged["y_pred"] - merged["y_pred_naive"]
    return merged


def _evaluate_normalized_cv_splits(
    setting_subset: pd.DataFrame,
    algorithm: str,
    rand_setting: str,
    test_mode: str,
) -> dict[str, dict]:
    eval_results_mod: dict[str, dict] = {}
    metric_names = list(AVAILABLE_METRICS.keys() - {"MAE", "MSE", "RMSE"})
    for cv_split in setting_subset["CV_split"].unique():
        cv_rows = setting_subset[setting_subset["CV_split"] == cv_split]
        dt = DrugResponseDataset(
            response=cv_rows["y_true"].to_numpy(),
            cell_line_ids=cv_rows["cell_line_name"].to_numpy(),
            drug_ids=cv_rows["drug_name"].to_numpy(),
            predictions=cv_rows["y_pred"].to_numpy(),
        )
        res = evaluate(dataset=dt, metric=metric_names)
        eval_results_mod[f"{algorithm}_{rand_setting}_{test_mode}_split_{cv_split}"] = res
    return eval_results_mod


def normalize_metrics_by_mean_effects(
    evaluation_results: pd.DataFrame,
    true_vs_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize metrics by subtracting NaiveMeanEffectsPredictor per setting.

    :param evaluation_results: Overall evaluation results table.
    :param true_vs_pred: True versus predicted values for all models.

    :returns: ``evaluation_results`` merged with ``: normalized`` metric columns.
    """
    naive_by_setting = _index_naive_predictions(true_vs_pred)
    eval_results_mod: dict[str, dict] = {}

    for algorithm in evaluation_results["algorithm"].unique():
        for rand_setting in evaluation_results["rand_setting"].unique():
            for test_mode in evaluation_results["test_mode"].unique():
                setting_subset = true_vs_pred[
                    (true_vs_pred["algorithm"] == algorithm)
                    & (true_vs_pred["rand_setting"] == rand_setting)
                    & (true_vs_pred["test_mode"] == test_mode)
                ]
                if setting_subset.empty:
                    continue
                naive_mean_effects = naive_by_setting[f"{test_mode}_{rand_setting}"]
                adjusted = _adjust_setting_subset(setting_subset, naive_mean_effects)
                eval_results_mod.update(_evaluate_normalized_cv_splits(adjusted, algorithm, rand_setting, test_mode))

    mod_table = pd.DataFrame.from_dict(eval_results_mod, orient="index")
    mod_table.columns = [f"{col}: normalized" for col in mod_table.columns]
    return evaluation_results.merge(mod_table, left_index=True, right_index=True)
