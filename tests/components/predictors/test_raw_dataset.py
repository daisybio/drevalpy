"""Tests for RawDatasetPredictor interface helpers."""

from __future__ import annotations

from drevalpy.components.predictors.base import Predictor
from drevalpy.components.predictors.raw_dataset import RawDatasetPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor


def test_raw_dataset_predictor_defaults() -> None:
    assert issubclass(RawDatasetPredictor, Predictor)
    assert RawDatasetPredictor.requires_drug_featurizer is False
    assert RawDatasetPredictor.required_cell_line_views == ()
    assert RawDatasetPredictor.required_drug_views == ()


def test_sparsego_active_views_follow_input_type() -> None:
    register_builtin_components()
    cls = get_predictor("sparsego")
    expression = cls(hyperparameters={"input_type": "expression"})
    mutations = cls(hyperparameters={"input_type": "mutations"})
    assert expression.active_cell_line_views() == ("gene_expression",)
    assert mutations.active_cell_line_views() == ("mutations",)
    assert cls.required_cell_line_views == ("gene_expression", "mutations")
