"""Tests for the shared Predictor constructor contract."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.base import Predictor
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import PredictorConfig


class _StubPredictor(Predictor):
    cell_line_contract: ClassVar[FeatureContract] = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    drug_contract: ClassVar[FeatureContract] = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
        return {"alpha": {"type": "float", "default": 1.0}}

    def fit(self, batch: ModelInputBatch) -> None:
        _ = batch

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


def test_predictor_init_merges_default_hyperparameters() -> None:
    predictor = _StubPredictor(hyperparameters={"alpha": 0.5, "extra": True})
    assert predictor._hyperparameters["alpha"] == 0.5
    assert predictor._hyperparameters["extra"] is True


def test_predictor_has_no_public_build() -> None:
    assert "build" not in Predictor.__dict__
    assert not hasattr(_StubPredictor(), "build")


def test_predictor_config_create_instance_passes_hyperparameters() -> None:
    register_builtin_components()
    predictor = PredictorConfig(name="elasticNet", hyperparameters={"alpha": 0.25}).create_instance()
    assert isinstance(predictor, ElasticNetPredictor)
    assert predictor._hyperparameters["alpha"] == 0.25
    assert predictor._h["alpha"] == 0.25
