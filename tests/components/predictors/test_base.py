"""Tests for the shared Predictor constructor contract."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
from drevalpy.components.core.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor
from drevalpy.components.registry.register_builtins import register_builtin_components
from drevalpy.models.config import PredictorConfig


class _StubPredictor(Predictor):
    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
        return {"alpha": {"type": "float", "default": 1.0}}

    def _fit(self, batch: ModelInputBatch) -> None:
        _ = batch

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


_StubPredictor.cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
_StubPredictor.drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_predictor_rejects_class_body_contracts() -> None:
    with pytest.raises(TypeError, match="do not set cell_line_contract"):

        class BadPredictor(Predictor):  # noqa: B903
            cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_predictor_init_merges_default_hyperparameters() -> None:
    predictor = _StubPredictor(hyperparameters={"alpha": 0.5, "extra": True})
    assert predictor._hyperparameters["alpha"] == 0.5
    assert predictor._hyperparameters["extra"] is True


def test_predictor_has_no_public_build() -> None:
    assert "build" not in Predictor.__dict__
    assert not hasattr(_StubPredictor(), "build")


def test_predictor_config_create_instance_passes_hyperparameters() -> None:
    register_builtin_components()
    predictor = PredictorConfig(name="elasticNet").create_instance({"alpha": 0.25})
    assert isinstance(predictor, ElasticNetPredictor)
    assert predictor._hyperparameters["alpha"] == 0.25
    assert predictor._h["alpha"] == 0.25
