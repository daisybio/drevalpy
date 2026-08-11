"""Tests for prep_results_format helpers."""

from __future__ import annotations

import pandas as pd

from drevalpy.visualization._legacy.prep_results_format import add_index_columns_from_model


def test_add_index_columns_from_model_splits_index() -> None:
    eval_results = pd.DataFrame({"MSE": [1.0]}, index=["Algo_predictions_LPO_split_0"])
    out = add_index_columns_from_model(eval_results)
    assert out.iloc[0]["algorithm"] == "Algo"
    assert out.iloc[0]["test_mode"] == "LPO"
