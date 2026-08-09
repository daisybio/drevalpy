"""Tests for the concrete DRPModel runtime."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)


def test_construct_model_supports_factory_lifecycle() -> None:
    elastic_net_cls = construct_model("ElasticNet")
    model = elastic_net_cls({"alpha": 0.1, "l1_ratio": 0.5})
    assert model.get_model_name() == "ElasticNet"
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape == (2,)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = elastic_net_cls.load(checkpoint)
        loaded_preds = loaded.predict(mudataset, split)
    assert np.allclose(preds, loaded_preds)


def test_naive_model_round_trip() -> None:
    naive_drug_mean_cls = construct_model("NaiveDrugMeanPredictor")
    model = naive_drug_mean_cls({})
    mudataset = synthetic_mudataset_identity()
    split = lco_split_masks()
    model.train(mudataset, split)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = naive_drug_mean_cls.load(checkpoint)
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()


def test_empty_training_transitions() -> None:
    import anndata as ad
    import pandas as pd

    import mudata as md
    from drevalpy.data.structures import SplitMasks
    from drevalpy.data.structures.dataset import Dataset

    naive_drug_mean_cls = construct_model("NaiveDrugMeanPredictor")
    model = naive_drug_mean_cls({})

    # All-NaN response matrix → empty training
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
        train=np.array([[0, 0], [0, 1]]),
        test=np.array([[1, 0], [1, 1]]),
        val=np.empty((0, 2), dtype=np.intp),
    )

    model.train(empty_mudataset, empty_split)
    assert model._empty_training is True
    empty_preds = model.predict(empty_mudataset, empty_split)
    assert np.isnan(empty_preds).all()

    mudataset = synthetic_mudataset_identity()
    split = lco_split_masks()
    model.train(mudataset, split)
    assert model._empty_training is False
    assert model._stack is not None
    assert model._stack.is_fitted()
    real_preds = model.predict(mudataset, split)
    assert np.isfinite(real_preds).all()

    model.train(empty_mudataset, empty_split)
    assert model._empty_training is True
    reempty_preds = model.predict(empty_mudataset, empty_split)
    assert np.isnan(reempty_preds).all()

    model.train(mudataset, split)
    assert model._empty_training is False
    again_preds = model.predict(mudataset, split)
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
    mudataset = synthetic_mudataset_identity()
    split = lco_split_masks()
    first = naive_drug_mean_cls()
    second = naive_drug_mean_cls()
    first.train(mudataset, split)
    assert first._stack is not None
    assert first._stack.is_fitted()
    assert second._stack is not None
    assert not second._stack.is_fitted()


def test_from_resolved_config_and_load_skip_default_stack() -> None:
    elastic_net_cls = construct_model("ElasticNet")
    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.2, "l1_ratio": 0.3})
    model = elastic_net_cls._from_resolved_config(config)
    assert model.hyperparameters["alpha"] == 0.2
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = elastic_net_cls.load(checkpoint)
    loaded_preds = loaded.predict(mudataset, split)
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
        model.hyperparameters = {"alpha": 0.25}  # type: ignore[misc]

    views = model.cell_line_views
    views.append("mutated_view")
    assert "mutated_view" not in model.cell_line_views
    with pytest.raises(AttributeError):
        model.cell_line_views = ["gene_expression"]  # type: ignore[misc]
    with pytest.raises(AttributeError):
        model.drug_views = ["fingerprints"]  # type: ignore[misc]

    assert not hasattr(model, "_sync_predictor_hyperparameters")
