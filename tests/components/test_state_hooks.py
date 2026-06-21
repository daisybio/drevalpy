"""Tests for component state hooks and DRP bridge sync."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import ElasticNet

from drevalpy.components.drp_bridge import (
    restore_naive_to_components,
    restore_sklearn_to_components,
    sync_naive_from_components,
    sync_sklearn_from_components,
)
from drevalpy.components.factory import naive_model_config
from drevalpy.components.predictors.naive import NaiveDrugMeanPredictor
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.baselines.naive_pred import NaiveDrugMeanPredictor as LegacyNaiveDrugMean
from drevalpy.models.baselines.sklearn_models import ElasticNetModel


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_sklearn_predictor_state_round_trip() -> None:
    predictor = ElasticNetPredictor()
    predictor.build({"alpha": 0.1}, {"cell_line": 2, "drug": 2, "n_classes": 1})
    predictor.fit(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1.0, 2.0]))
    state = predictor.get_state()
    restored = ElasticNetPredictor()
    restored.set_state(state)
    assert restored.is_fitted()
    assert np.allclose(restored.predict(np.array([[0.5, 0.5]])), predictor.predict(np.array([[0.5, 0.5]])))


def test_naive_predictor_state_round_trip() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    composed = naive_model_config("naiveDrugMean").create_model()
    composed.train(response, FeatureDataset(features={}), None)
    predictor = composed._predictor
    state = predictor.get_state()
    restored = NaiveDrugMeanPredictor()
    restored.set_state(state)
    assert restored.is_fitted()
    assert restored._entity_means == predictor._entity_means


def test_sklearn_bridge_sync_and_restore() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )
    legacy = ElasticNetModel()
    legacy.build_model(hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    legacy.train(response, cell_line_input, drug_input)
    sync_sklearn_from_components(legacy)
    assert legacy.model is not None
    assert legacy.gene_expression_scaler is not None

    legacy.model = ElasticNet(alpha=0.2)
    legacy.gene_expression_scaler.mean_ = np.array([0.0, 0.0, 0.0])
    restore_sklearn_to_components(legacy)
    composed = legacy._component_bridge.composed
    assert composed is not None
    assert composed._predictor.get_state()["estimator"] is legacy.model


def test_naive_bridge_sync_and_restore() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    legacy = LegacyNaiveDrugMean()
    legacy.build_model(hyperparameters={})
    legacy.train(response, FeatureDataset(features={}), None)
    sync_naive_from_components(legacy, "naiveDrugMean")
    assert legacy.drug_means["d1"] == pytest.approx(1.0)
    legacy.drug_means["d1"] = 99.0
    restore_naive_to_components(legacy, "naiveDrugMean")
    composed = legacy._component_bridge.composed
    assert composed is not None
    assert composed._predictor.get_state()["drug_means"]["d1"] == 99.0
