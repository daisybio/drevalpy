"""Root public contract tests for MODEL_FACTORY and named facades."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import (
    MODEL_FACTORY,
    MULTI_DRUG_MODEL_FACTORY,
    SINGLE_DRUG_MODEL_FACTORY,
    ElasticNetModel,
    NaivePredictor,
    RandomForest,
    SingleDrugElasticNet,
)
from drevalpy.models._factory_classes import SYMBOL_TO_FACTORY_NAME, symbol_for_factory_name
from drevalpy.models._native_drp_model import NativeDRPModel
from drevalpy.models.zoo import list_zoo_names
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)

_FAST_EXECUTION_MODELS = (
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "ElasticNet",
    "RandomForest",
    "SingleDrugElasticNet",
)

_SINGLE_DRUG_NAMES = frozenset({"SingleDrugElasticNet", "SingleDrugRandomForest", "MOLIR", "SuperFELTR"})
_EARLY_STOPPING_NAMES = frozenset(
    {
        "DIPK",
        "MOLIR",
        "MultiViewNeuralNetwork",
        "PharmaFormer",
        "SimpleNeuralNetwork",
        "SuperFELTR",
    }
)


def _minimal_hyperparameters(model_name: str) -> dict:
    model = MODEL_FACTORY[model_name]()
    hyperparameters = model.get_hyperparameter_set()[0]
    if model_name == "DIPK":
        return {**hyperparameters, "epochs": 1, "epochs_autoencoder": 1, "heads": 1}
    if model_name in {"SimpleNeuralNetwork", "MultiViewNeuralNetwork"}:
        return {**hyperparameters, "units_per_layer": [2, 2], "max_epochs": 1}
    if model_name == "PharmaFormer":
        return {**hyperparameters, "epochs": 1, "patience": 2}
    if model_name == "Precily":
        return {**hyperparameters, "epochs": 1, "batch_size": 32}
    return hyperparameters


def test_factory_names_match_zoo_and_subsets() -> None:
    factory_names = set(MODEL_FACTORY)
    assert factory_names == set(list_zoo_names(include_external=False))
    assert set(SINGLE_DRUG_MODEL_FACTORY) == _SINGLE_DRUG_NAMES
    assert set(MULTI_DRUG_MODEL_FACTORY) == factory_names - _SINGLE_DRUG_NAMES
    assert set(SINGLE_DRUG_MODEL_FACTORY).isdisjoint(MULTI_DRUG_MODEL_FACTORY)
    assert {**MULTI_DRUG_MODEL_FACTORY, **SINGLE_DRUG_MODEL_FACTORY} == MODEL_FACTORY


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORY))
def test_factory_entry_is_native_facade(model_name: str) -> None:
    model_cls = MODEL_FACTORY[model_name]
    assert issubclass(model_cls, NativeDRPModel)
    assert model_cls.get_model_name() == model_name
    assert model_cls.is_single_drug_model is (model_name in _SINGLE_DRUG_NAMES)
    assert model_cls.early_stopping is (model_name in _EARLY_STOPPING_NAMES)
    symbol = symbol_for_factory_name(model_name)
    assert SYMBOL_TO_FACTORY_NAME.get(symbol, symbol) == model_name


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORY))
def test_factory_entry_builds_flat_hyperparameters(model_name: str) -> None:
    model = MODEL_FACTORY[model_name]()
    defaults = model.get_default_hyperparameters()
    assert isinstance(defaults, dict)
    space = model.get_structured_hyperparameter_space()
    assert isinstance(space, dict)
    try:
        model.build_model(_minimal_hyperparameters(model_name))
    except ImportError as exc:
        if model_name in {"MultiViewXGBoost", "MultiViewLightGBM"}:
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize("model_name", _FAST_EXECUTION_MODELS)
def test_fast_execution_matrix_train_predict_save_load(model_name: str) -> None:
    response = multi_drug_response()
    if model_name.startswith("Naive"):
        cell_line_input = identity_cell_line_features()
        drug_input: FeatureDataset | None = identity_drug_features()
    else:
        cell_line_input = cell_line_gene_expression()
        drug_input = drug_fingerprints()
    if model_name.startswith("SingleDrug"):
        drug_input = None

    model = MODEL_FACTORY[model_name]()
    model.build_model(_minimal_hyperparameters(model_name))
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = MODEL_FACTORY[model_name].load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds, equal_nan=True)


def test_named_root_exports_match_factory() -> None:
    assert NaivePredictor is MODEL_FACTORY["NaivePredictor"]
    assert ElasticNetModel is MODEL_FACTORY["ElasticNet"]
    assert RandomForest is MODEL_FACTORY["RandomForest"]
    assert SingleDrugElasticNet is MODEL_FACTORY["SingleDrugElasticNet"]


def test_empty_training_predicts_nan() -> None:
    model = NaivePredictor()
    model.build_model({})
    empty = DrugResponseDataset(
        response=np.array([]),
        cell_line_ids=np.array([]),
        drug_ids=np.array([]),
    )
    model.train(empty, identity_cell_line_features(), identity_drug_features())
    preds = model.predict(
        np.array(["cl1"]),
        np.array(["d1"]),
        identity_cell_line_features(),
        identity_drug_features(),
    )
    assert preds.shape == (1,)
    assert np.isnan(preds).all()
