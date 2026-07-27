"""Tests for the shared NativeDRPModel facade."""

from __future__ import annotations

import tempfile

import numpy as np

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


def test_sync_predictor_hyperparameters_updates_composed_config() -> None:
    NativeElasticNet = create_native_drp_class("ElasticNet", spec="ElasticNet", validate_spec=False)
    model = NativeElasticNet({"alpha": 0.1, "l1_ratio": 0.5})
    assert model._composed is not None
    model.hyperparameters["alpha"] = 0.25
    model._sync_predictor_hyperparameters()
    assert model._composed.config is not None
    assert model._composed.config.predictor.hyperparameters["alpha"] == 0.25
    assert model._resolved_model_config is not None
    assert model._resolved_model_config.predictor.hyperparameters["alpha"] == 0.25
