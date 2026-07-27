"""Tests for the concrete DRPModel runtime."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)


def test_construct_model_supports_factory_lifecycle() -> None:
    ElasticNet = construct_model("ElasticNet")
    model = ElasticNet({"alpha": 0.1, "l1_ratio": 0.5})
    assert model.get_model_name() == "ElasticNet"
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = drug_fingerprints()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = ElasticNet.load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


def test_naive_model_round_trip() -> None:
    NaiveDrugMean = construct_model("NaiveDrugMeanPredictor")
    model = NaiveDrugMean({})
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    model.train(response, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = NaiveDrugMean.load(tmp)
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()


def test_empty_training_transitions() -> None:
    NaiveDrugMean = construct_model("NaiveDrugMeanPredictor")
    model = NaiveDrugMean({})
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
    assert model._stack is not None
    assert model._stack.is_fitted()
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
    ElasticNet = construct_model("ElasticNet")
    model = ElasticNet()
    assert model.hyperparameters == ElasticNet.get_default_hyperparameters()
    assert not hasattr(model, "configure")
    assert not hasattr(ElasticNet, "configure")


def test_constructor_overrides_affect_views_before_feature_load() -> None:
    RandomForest = construct_model("RandomForest")
    defaults = RandomForest()
    overridden = RandomForest({"n_estimators": 3, "max_depth": 2})
    assert overridden.hyperparameters["n_estimators"] == 3
    assert overridden.cell_line_views == defaults.cell_line_views
    assert overridden.drug_views == defaults.drug_views
    assert overridden._stack is not None
    assert overridden._stack.config is not None
    assert overridden._stack.config.predictor.hyperparameters["n_estimators"] == 3


def test_separate_constructor_calls_have_isolated_fitted_state() -> None:
    NaiveDrugMean = construct_model("NaiveDrugMeanPredictor")
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    first = NaiveDrugMean()
    second = NaiveDrugMean()
    first.train(response, cell_line_input, drug_input)
    assert first._stack is not None
    assert first._stack.is_fitted()
    assert second._stack is not None
    assert not second._stack.is_fitted()


def test_from_resolved_config_and_load_skip_default_stack() -> None:
    ElasticNet = construct_model("ElasticNet")
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.2, "l1_ratio": 0.3})
    model = ElasticNet._from_resolved_config(config)
    assert model.hyperparameters["alpha"] == 0.2
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = drug_fingerprints()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = ElasticNet.load(tmp)
    loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()


def test_hyperparameters_and_views_are_immutable_after_construction() -> None:
    ElasticNet = construct_model("ElasticNet")
    model = ElasticNet({"alpha": 0.1, "l1_ratio": 0.5})
    assert model._stack is not None

    exposed = model.hyperparameters
    exposed["alpha"] = 0.25
    assert model.hyperparameters["alpha"] == 0.1
    assert model._stack.config is not None
    assert model._stack.config.predictor.hyperparameters["alpha"] == 0.1

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


def test_load_drug_features_stores_preload_without_mutating_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drevalpy.components.predictors.literature.structured_engine_adapter import (
        DISCOVERED_HYPERPARAMETERS_KEY,
    )
    from drevalpy.components.registry import get_predictor
    from drevalpy.datasets.dataset import FeatureDataset

    ElasticNet = construct_model("ElasticNet")
    model = ElasticNet({"alpha": 0.1, "l1_ratio": 0.5})
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
