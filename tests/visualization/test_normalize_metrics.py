"""Tests for normalize_metrics."""

from __future__ import annotations

import pandas as pd

from drevalpy.visualization._legacy.normalize_metrics import normalize_metrics_by_mean_effects


def test_normalize_metrics_adds_normalized_columns() -> None:
    eval_results = pd.DataFrame({"MSE": [1.0]}, index=["Algo_predictions_LPO_split_0"])
    eval_results["algorithm"] = ["Algo"]
    eval_results["rand_setting"] = ["predictions"]
    eval_results["test_mode"] = ["LPO"]
    eval_results["CV_split"] = ["0"]

    t_vs_p = pd.DataFrame(
        {
            "algorithm": ["Algo", "NaiveMeanEffectsPredictor"],
            "rand_setting": ["predictions", "predictions"],
            "test_mode": ["LPO", "LPO"],
            "CV_split": ["0", "0"],
            "drug_name": ["d1", "d1"],
            "cell_line_name": ["c1", "c1"],
            "y_true": [1.0, 1.0],
            "y_pred": [0.5, 0.2],
        }
    )
    out = normalize_metrics_by_mean_effects(eval_results, t_vs_p)
    assert any(": normalized" in col for col in out.columns)
