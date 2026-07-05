"""Tests for component-native state and legacy checkpoint loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import ElasticNet

from drevalpy.components.predictors.naive import NaiveDrugMeanPredictor
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import restore_naive_to_components, restore_sklearn_to_components
from drevalpy.models._legacy_checkpoint_loaders import load_legacy_naive_checkpoint, load_legacy_sklearn_checkpoint
from drevalpy.models.baselines.naive_pred import NaiveDrugMeanPredictor as LegacyNaiveDrugMean
from drevalpy.models.baselines.sklearn_models import ElasticNetModel
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.legacy_checkpoint_migration import migrate_checkpoint_to_component_stack


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
    composed = model_config_for_name("NaiveDrugMeanPredictor").create_model()
    composed.train(response, FeatureDataset(features={}), FeatureDataset(features={}))
    predictor = composed._predictor
    state = predictor.get_state()
    restored = NaiveDrugMeanPredictor()
    restored.set_state(state)
    assert restored.is_fitted()
    assert restored._entity_means == predictor._entity_means


def test_sklearn_native_save_load_round_trip() -> None:
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
    model = ElasticNetModel()
    model.build_model(hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    model.train(response, cell_line_input, drug_input)
    assert model.model is not None
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = ElasticNetModel.load(tmp)
        preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input))


def test_naive_native_save_load_round_trip() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    model = LegacyNaiveDrugMean()
    model.build_model(hyperparameters={})
    model.train(response, FeatureDataset(features={}), None)
    assert model.drug_means["d1"] == pytest.approx(1.0)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = LegacyNaiveDrugMean.load(tmp)
    assert loaded.drug_means["d1"] == pytest.approx(1.0)


def test_sklearn_legacy_checkpoint_load() -> None:
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
    model = ElasticNetModel()
    model.build_model(hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    model.train(response, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        legacy_dir = Path(tmp) / "legacy"
        legacy_dir.mkdir()
        import joblib

        joblib.dump(ElasticNet(alpha=0.2), legacy_dir / "model.pkl")
        with open(legacy_dir / "hyperparameters.json", "w") as handle:
            json.dump({"alpha": 0.2, "l1_ratio": 0.5}, handle)
        loaded = ElasticNetModel()
        load_legacy_sklearn_checkpoint(loaded, str(legacy_dir))
        composed = loaded._component_bridge.composed
        assert composed is not None
        assert composed._predictor.get_state()["estimator"] is not None


def test_naive_legacy_checkpoint_restore() -> None:
    model = LegacyNaiveDrugMean()
    model.build_model({})
    object.__setattr__(model, "_legacy_dataset_mean", 1.5)
    object.__setattr__(model, "_legacy_drug_means", {"d1": 1.0, "d2": 2.0})
    restore_naive_to_components(model, "naiveDrugMean")
    composed = model._component_bridge.composed
    assert composed is not None
    assert composed._predictor.get_state()["drug_means"]["d1"] == 1.0


def test_migrate_checkpoint_to_component_stack() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    model = LegacyNaiveDrugMean()
    model.build_model({})
    model.train(response, FeatureDataset(features={}), None)
    with tempfile.TemporaryDirectory() as tmp:
        native_dir = Path(tmp) / "native"
        native_dir.mkdir()
        migrate_checkpoint_to_component_stack(model, str(tmp), output_directory=str(native_dir))
        assert (native_dir / "component_stack.joblib").exists()
