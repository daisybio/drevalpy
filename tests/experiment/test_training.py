"""Tests for experiment_training.train_and_predict_impl."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.experiment.training import train_and_predict_impl
from drevalpy.models._model_lookup import get_model_class


def test_train_and_predict_requires_dataset_name() -> None:
    model = get_model_class("NaiveMeanEffectsPredictor")()
    train = DrugResponseDataset(
        response=np.array([1.0]),
        cell_line_ids=np.array(["A"]),
        drug_ids=np.array(["d"]),
        dataset_name=None,
    )
    test = train.copy()
    with pytest.raises(ValueError, match="dataset_name"):
        train_and_predict_impl(model, "data", train, test)
