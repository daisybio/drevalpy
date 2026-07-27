"""Tests for experiment_consolidate."""

from __future__ import annotations

import pandas as pd

from drevalpy.experiment.consolidate import (
    consolidate_single_drug_model_predictions_impl,
)
from drevalpy.models._model_lookup import get_model_class


def test_consolidate_single_drug_merges_per_drug_csvs(tmp_path) -> None:
    model_class = get_model_class("MOLIR")
    model_name = model_class.get_model_name()
    results = tmp_path / "results"
    for drug in ("DrugA", "DrugB"):
        pred_dir = results / model_name / "drugs" / drug / "predictions"
        pred_dir.mkdir(parents=True)
        pd.DataFrame({"response": [1.0], "predictions": [1.1]}).to_csv(pred_dir / "predictions_split_0.csv")

    consolidate_single_drug_model_predictions_impl(
        models=[model_class],
        n_cv_splits=1,
        results_path=str(results),
        cross_study_datasets=[],
        out_path=str(results),
    )

    out_file = results / model_name / "predictions" / "predictions_split_0.csv"
    assert out_file.is_file()
    merged = pd.read_csv(out_file, index_col=0)
    assert len(merged) == 2
