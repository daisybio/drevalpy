"""Tests for the shared NativeDRPModel facade."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models._native_drp_model import create_native_drp_class
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)


def test_native_drp_class_supports_factory_lifecycle() -> None:
    NativeElasticNet = create_native_drp_class("ElasticNet", spec="ElasticNet", validate_spec=False)
    model = NativeElasticNet({"alpha": 0.1, "l1_ratio": 0.5})
    assert model.get_model_name() == "ElasticNet"
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = drug_fingerprints()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = NativeElasticNet.load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


def test_native_naive_class_round_trip() -> None:
    NativeNaive = create_native_drp_class("NaiveDrugMeanPredictor", spec="NaiveDrugMeanPredictor", validate_spec=False)
    model = NativeNaive({})
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    model.train(response, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = NativeNaive.load(tmp)
    assert loaded._composed is not None
    assert loaded._composed.is_fitted()


def test_empty_training_transitions() -> None:
    NativeNaive = create_native_drp_class("NaiveDrugMeanPredictor", spec="NaiveDrugMeanPredictor", validate_spec=False)
    model = NativeNaive({})
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    empty = DrugResponseDataset(
        response=np.array([]),
        cell_line_ids=np.array([]),
        drug_ids=np.array([]),
    )
    response = multi_drug_response()

    model.train(empty, cell_line_input, drug_input)
    assert model._empty_training is True
    empty_preds = model.predict(
        np.array(["cl1"]),
        np.array(["d1"]),
        cell_line_input,
        drug_input,
    )
    assert np.isnan(empty_preds).all()

    model.train(response, cell_line_input, drug_input)
    assert model._empty_training is False
    assert model._composed is not None
    assert model._composed.is_fitted()
    real_preds = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )
    assert np.isfinite(real_preds).all()

    model.train(empty, cell_line_input, drug_input)
    assert model._empty_training is True
    reempty_preds = model.predict(
        np.array(["cl1"]),
        np.array(["d1"]),
        cell_line_input,
        drug_input,
    )
    assert np.isnan(reempty_preds).all()

    model.train(response, cell_line_input, drug_input)
    assert model._empty_training is False
    again_preds = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )
    assert np.isfinite(again_preds).all()


def test_constructor_defaults_match_classmethod() -> None:
    NativeElasticNet = create_native_drp_class("ElasticNet", spec="ElasticNet", validate_spec=False)
    model = NativeElasticNet()
    assert model.hyperparameters == NativeElasticNet.get_default_hyperparameters()
    assert not hasattr(model, "configure")
    assert not hasattr(NativeElasticNet, "configure")


def test_constructor_overrides_affect_views_before_feature_load() -> None:
    NativeRF = create_native_drp_class("RandomForest", spec="RandomForest", validate_spec=False)
    defaults = NativeRF()
    overridden = NativeRF({"n_estimators": 3, "max_depth": 2})
    assert overridden.hyperparameters["n_estimators"] == 3
    assert overridden.cell_line_views == defaults.cell_line_views
    assert overridden.drug_views == defaults.drug_views
    assert overridden._composed is not None
    assert overridden._composed.config is not None
    assert overridden._composed.config.predictor.hyperparameters["n_estimators"] == 3


def test_separate_constructor_calls_have_isolated_fitted_state() -> None:
    NativeNaive = create_native_drp_class("NaiveDrugMeanPredictor", spec="NaiveDrugMeanPredictor", validate_spec=False)
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    first = NativeNaive()
    second = NativeNaive()
    first.train(response, cell_line_input, drug_input)
    assert first._composed is not None
    assert first._composed.is_fitted()
    assert second._composed is not None
    assert not second._composed.is_fitted()


def test_from_model_config_and_load_skip_default_stack() -> None:
    from drevalpy.models.config import ModelConfig

    NativeElasticNet = create_native_drp_class("ElasticNet", spec="ElasticNet", validate_spec=False)
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.2, "l1_ratio": 0.3})
    model = NativeElasticNet.from_model_config(config)
    assert model.hyperparameters["alpha"] == 0.2
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = drug_fingerprints()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = NativeElasticNet.load(tmp)
    loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)
    assert loaded._composed is not None
    assert loaded._composed.is_fitted()


def test_facade_hyperparameters_and_views_are_immutable_after_construction() -> None:
    NativeElasticNet = create_native_drp_class("ElasticNet", spec="ElasticNet", validate_spec=False)
    model = NativeElasticNet({"alpha": 0.1, "l1_ratio": 0.5})
    assert model._composed is not None

    exposed = model.hyperparameters
    exposed["alpha"] = 0.25
    assert model.hyperparameters["alpha"] == 0.1
    assert model._composed.config is not None
    assert model._composed.config.predictor.hyperparameters["alpha"] == 0.1

    with pytest.raises(AttributeError):
        model.hyperparameters = {"alpha": 0.25}  # type: ignore[misc]

    views = model.cell_line_views
    views.append("mutated_view")
    assert "mutated_view" not in model.cell_line_views
    with pytest.raises(AttributeError):
        model.cell_line_views = ["gene_expression"]  # type: ignore[misc]
    with pytest.raises(AttributeError):
        model.drug_views = ["fingerprints"]  # type: ignore[misc]

    assert not hasattr(model, "_sync_predictor_hyperparameters")


def test_load_drug_features_stores_preload_without_mutating_facade_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drevalpy.components.predictors.literature.structured_engine_adapter import (
        DISCOVERED_HYPERPARAMETERS_KEY,
    )
    from drevalpy.components.registry import get_predictor
    from drevalpy.datasets.dataset import FeatureDataset

    NativeElasticNet = create_native_drp_class("ElasticNet", spec="ElasticNet", validate_spec=False)
    model = NativeElasticNet({"alpha": 0.1, "l1_ratio": 0.5})
    original = model.hyperparameters
    assert model._resolved_model_config is not None
    predictor_cls = get_predictor(model._resolved_model_config.predictor.name)

    def _fake_loader(
        cls: type,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict | None = None,
        model_name: str | None = None,
    ):
        _ = cls, data_path, dataset_name, model_name
        assert hyperparameters is not None
        hyperparameters["alpha"] = 999.0
        return (
            FeatureDataset(features={"d1": {"fingerprints": np.array([1.0])}}),
            {DISCOVERED_HYPERPARAMETERS_KEY: {"drug_dim": 64}},
        )

    monkeypatch.setattr(
        predictor_cls,
        "load_dataset_drug_features",
        classmethod(_fake_loader),
        raising=False,
    )

    features = model.load_drug_features(".", "TOY")
    assert features is not None
    assert model.hyperparameters == original
    assert model._engine_preload_state[DISCOVERED_HYPERPARAMETERS_KEY] == {"drug_dim": 64}
