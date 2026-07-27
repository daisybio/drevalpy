"""Tests for SuperFELTR training orchestration."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.literature.impl.superfeltr.superfeltr import SuperFELTR
from drevalpy.components.predictors.literature.impl.superfeltr.training import run_superfeltr_training
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


def test_run_superfeltr_training_skips_when_output_empty() -> None:
    model = SuperFELTR()
    model.configure(SuperFELTR.get_default_hyperparameters())
    empty_output = DrugResponseDataset(
        response=np.array([]),
        cell_line_ids=np.array([]),
        drug_ids=np.array([]),
    )
    cell_line_input = FeatureDataset(features={})

    run_superfeltr_training(model, empty_output, cell_line_input, None, "checkpoints")

    assert model.expr_encoder is None
    assert model.regressor is None
    assert model.best_checkpoint is None
