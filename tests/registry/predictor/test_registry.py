"""Tests for PredictorRegistry type and singleton."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import PredictorRegistry, predictor_registry
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


def test_predictor_registry_uses_fixed_identity() -> None:
    registry = PredictorRegistry()
    assert registry._registry_id == "predictor"
    assert predictor_registry._display_name == "predictors"


def test_isolated_predictor_registry_registers_with_contracts() -> None:
    registry = PredictorRegistry()

    @registry.register(
        "localPred",
        description="local",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.GRAPH,
    )
    class LocalPred(FeatureFreePredictor):
        def _fit(self, batch: ModelInputBatch) -> None:
            return None

        def _predict(self, batch: ModelInputBatch) -> np.ndarray:
            return np.zeros(batch.n_pairs, dtype=np.float64)

    assert registry.get("localPred") is LocalPred
    assert vars(LocalPred)["cell_line_contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert vars(LocalPred)["drug_contract"] == FeatureContract(format=FeatureFormat.GRAPH)
