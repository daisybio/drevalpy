"""Tests for PredictorRegistry type and singleton."""

from __future__ import annotations

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.registry.predictor_registry import PredictorRegistry, predictor_registry


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
        pass

    assert registry.get("localPred") is LocalPred
    assert vars(LocalPred)["cell_line_contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert vars(LocalPred)["drug_contract"] == FeatureContract(format=FeatureFormat.GRAPH)
