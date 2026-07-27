"""Tests for experiment_final_model helpers."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.experiment.final_model import _prepare_final_train_val
from drevalpy.models._model_lookup import get_model_class


def test_prepare_final_train_val_splits_and_removes_nan() -> None:
    model_class = get_model_class("NaivePredictor")
    full = DrugResponseDataset(
        response=np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        cell_line_ids=np.array([f"CL{i}" for i in range(10)]),
        drug_ids=np.array(["d"] * 10),
        dataset_name="Toy_Data",
    )
    train, validation, early_stopping = _prepare_final_train_val(
        full.copy(),
        test_mode="LCO",
        val_ratio=0.2,
        model_class=model_class,
    )
    assert len(train) + len(validation) == 9
    assert early_stopping is None
