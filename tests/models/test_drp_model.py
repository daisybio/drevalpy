"""Tests for the concrete DRPModel runtime."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)


def test_construct_model_supports_factory_lifecycle() -> None:
    elastic_net_cls = construct_model("ElasticNet")
    model = elastic_net_cls({"alpha": 0.1, "l1_ratio": 0.5})
    assert model.get_model_name() == "ElasticNet"
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = drug_fingerprints()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = elastic_net_cls.load(checkpoint)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)


def test_naive_model_round_trip() -> None:
    naive_drug_mean_cls = construct_model("NaiveDrugMeanPredictor")
    model = naive_drug_mean_cls({})
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    model.train(response, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = naive_drug_mean_cls.load(checkpoint)
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()


def test_empty_training_transitions() -> None:
    naive_drug_mean_cls = construct_model("NaiveDrugMeanPredictor")
    model = naive_drug_mean_cls({})
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
    elastic_net_cls = construct_model("ElasticNet")
    model = elastic_net_cls()
    assert model.hyperparameters == elastic_net_cls.get_default_hyperparameters()
    assert not hasattr(model, "configure")
    assert not hasattr(elastic_net_cls, "configure")


def test_constructor_overrides_affect_views_before_feature_load() -> None:
    random_forest_cls = construct_model("RandomForest")
    defaults = random_forest_cls()
    overridden = random_forest_cls({"n_estimators": 3, "max_depth": 2})
    assert overridden.hyperparameters["n_estimators"] == 3
    assert overridden.cell_line_views == defaults.cell_line_views
    assert overridden.drug_views == defaults.drug_views
    assert overridden._stack is not None
    assert overridden._resolved_model_config is not None
    assert overridden._resolved_model_config.predictor_values()["n_estimators"] == 3


def test_separate_constructor_calls_have_isolated_fitted_state() -> None:
    naive_drug_mean_cls = construct_model("NaiveDrugMeanPredictor")
    response = multi_drug_response()
    cell_line_input = identity_cell_line_features()
    drug_input = identity_drug_features()
    first = naive_drug_mean_cls()
    second = naive_drug_mean_cls()
    first.train(response, cell_line_input, drug_input)
    assert first._stack is not None
    assert first._stack.is_fitted()
    assert second._stack is not None
    assert not second._stack.is_fitted()


def test_from_resolved_config_and_load_skip_default_stack() -> None:
    elastic_net_cls = construct_model("ElasticNet")
    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.2, "l1_ratio": 0.3})
    model = elastic_net_cls._from_resolved_config(config)
    assert model.hyperparameters["alpha"] == 0.2
    response = multi_drug_response()
    cell_line_input = cell_line_gene_expression()
    drug_input = drug_fingerprints()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = elastic_net_cls.load(checkpoint)
    loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds)
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()


def test_hyperparameters_and_views_are_immutable_after_construction() -> None:
    elastic_net_cls = construct_model("ElasticNet")
    model = elastic_net_cls({"alpha": 0.1, "l1_ratio": 0.5})
    assert model._stack is not None

    exposed = model.hyperparameters
    exposed["alpha"] = 0.25
    assert model.hyperparameters["alpha"] == 0.1
    assert model._resolved_model_config is not None
    assert model._resolved_model_config.predictor_values()["alpha"] == 0.1

    with pytest.raises(AttributeError):
        model.hyperparameters = {"alpha": 0.25}

    views = model.cell_line_views
    views.append("mutated_view")
    assert "mutated_view" not in model.cell_line_views
    with pytest.raises(AttributeError):
        model.cell_line_views = ["gene_expression"]
    with pytest.raises(AttributeError):
        model.drug_views = ["fingerprints"]

    assert not hasattr(model, "_sync_predictor_hyperparameters")


def test_load_drug_features_uses_featurizer_loader_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drevalpy.components.registry import get_predictor
    from drevalpy.datasets.dataset import FeatureDataset

    elastic_net_cls = construct_model("ElasticNet")
    model = elastic_net_cls({"alpha": 0.1, "l1_ratio": 0.5})
    assert model._resolved_model_config is not None
    predictor_cls = get_predictor(model._resolved_model_config.template.predictor.name)

    def _fake_loader(
        cls: type,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict | None = None,
        model_name: str | None = None,
    ):
        _ = cls, data_path, dataset_name, model_name
        _ = hyperparameters
        raise AssertionError("predictor loader must not run when a featurizer is configured")

    monkeypatch.setattr(
        predictor_cls,
        "load_dataset_drug_features",
        classmethod(_fake_loader),
        raising=False,
    )

    expected = FeatureDataset(features={"d1": {"fingerprints": np.array([1.0])}})
    monkeypatch.setattr(
        "drevalpy.components.data_loading.load_drug_features_for_model_config",
        lambda *args, **kwargs: expected,
    )
    assert model.load_drug_features(".", "TOY") is expected
