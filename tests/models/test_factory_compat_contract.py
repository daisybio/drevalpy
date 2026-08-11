"""Root public contract tests for MODEL_FACTORY and construct_model."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.models import MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY, construct_model
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.zoo import list_zoo_names
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)

_FAST_EXECUTION_MODELS = (
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "ElasticNet",
    "RandomForest",
    "SingleDrugElasticNet",
)

# Dependency-light build coverage (nox tests session, core-only install phase).
_LIGHT_BUILD_MODELS = frozenset(
    {
        "NaivePredictor",
        "NaiveDrugMeanPredictor",
        "NaiveCellLineMeanPredictor",
        "NaiveTissueMeanPredictor",
        "NaiveTissueDrugMeanPredictor",
        "NaiveMeanEffectsPredictor",
        "ElasticNet",
        "Lasso",
        "RandomForest",
        "SVR",
        "GradientBoosting",
        "AdaBoostDecisionTree",
        "KNNRegressor",
        "SingleDrugElasticNet",
        "SingleDrugRandomForest",
        "MultiViewRandomForest",
    }
)

_OPTIONAL_EXTRA_MODELS = frozenset(
    {
        "MultiViewXGBoost",
        "MultiViewLightGBM",
        "Precily",
        "SparseGO",
    }
)

_SINGLE_DRUG_NAMES = frozenset({"SingleDrugElasticNet", "SingleDrugRandomForest", "MOLIR", "SuperFELTR"})


def _predictor_supports_early_stopping(model_name: str) -> bool:
    from drevalpy.components.registry import get_predictor
    from drevalpy.models.zoo import get_zoo_config

    try:
        predictor_cls = get_predictor(get_zoo_config(model_name).predictor.name)
    except ImportError:
        return False
    return bool(getattr(predictor_cls, "supports_early_stopping", False))


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
def test_factory_entry_is_drp_model(model_name: str) -> None:
    model_cls = MODEL_FACTORY[model_name]
    assert issubclass(model_cls, DRPModel)
    assert model_cls.get_model_name() == model_name
    assert model_cls.is_single_drug() is (model_name in _SINGLE_DRUG_NAMES)
    assert model_cls.supports_early_stopping() is _predictor_supports_early_stopping(model_name)


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORY))
def test_construct_model_matches_factory(model_name: str) -> None:
    assert construct_model(model_name) is MODEL_FACTORY[model_name]


@pytest.mark.parametrize("model_name", sorted(_LIGHT_BUILD_MODELS))
def test_factory_entry_builds_flat_hyperparameters_light(model_name: str) -> None:
    model_cls = MODEL_FACTORY[model_name]
    defaults = model_cls.get_default_hyperparameters()
    assert isinstance(defaults, dict)
    space = model_cls.get_structured_hyperparameter_space()
    assert isinstance(space, dict)
    model_cls(_minimal_hyperparameters(model_name))


@pytest.mark.parametrize("model_name", sorted(set(MODEL_FACTORY) - _LIGHT_BUILD_MODELS))
def test_factory_entry_builds_flat_hyperparameters_full(model_name: str) -> None:
    # Full-suite build coverage for literature / optional-extra models.
    model_cls = MODEL_FACTORY[model_name]
    defaults = model_cls.get_default_hyperparameters()
    assert isinstance(defaults, dict)
    space = model_cls.get_structured_hyperparameter_space()
    assert isinstance(space, dict)
    try:
        model_cls(_minimal_hyperparameters(model_name))
    except ImportError as exc:
        if model_name in _OPTIONAL_EXTRA_MODELS:
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize("model_name", _FAST_EXECUTION_MODELS)
def test_fast_execution_matrix_train_predict_save_load(model_name: str) -> None:
    if model_name.startswith("Naive"):
        mudataset = synthetic_mudataset_identity()
    else:
        mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()

    model = MODEL_FACTORY[model_name](_minimal_hyperparameters(model_name))
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = MODEL_FACTORY[model_name].load(checkpoint)
        loaded_preds = loaded.predict(mudataset, split)
    assert np.allclose(preds, loaded_preds, equal_nan=True)


def test_empty_training_predicts_nan() -> None:
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types import SplitMask, SplitMasks
    from drevalpy.types.data.dataset import Dataset

    model = construct_model("NaivePredictor")({})

    # All-NaN matrix produces empty training
    nan_response = np.full((2, 2), np.nan, dtype=np.float32)
    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])
    empty_ad = ad.AnnData(
        X=nan_response,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["L", "B"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    empty_mudataset = Dataset(md.MuData({"response": empty_ad}), name="test")
    empty_split = SplitMasks(
        train=SplitMask(np.array([[True, True], [False, False]])),
        test=SplitMask(np.array([[False, False], [True, True]])),
        val=SplitMask(np.zeros((2, 2), dtype=bool)),
    )

    model.train(empty_mudataset, empty_split)
    preds = model.predict(empty_mudataset, empty_split)
    assert np.isnan(preds).all()
