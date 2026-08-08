"""construct_model vs ModelConfig._from_resolved_config parity."""

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

_PARITY_CASES = (
    ("NaivePredictor", "factory"),
    ("NaiveDrugMeanPredictor", "factory"),
    ("ElasticNet", "factory"),
    ("RandomForest", "factory"),
    ("PcaIdentityRF", "construct_model"),
)


def _model_class(model_name: str, entrypoint: str):
    if entrypoint == "construct_model":
        return construct_model(model_name, "pca[expression]:identity:randomForest")
    return construct_model(model_name)


def _minimal_hp(model_name: str) -> dict:
    if model_name.startswith("Naive"):
        return {}
    if model_name == "ElasticNet":
        return {"alpha": 0.1, "l1_ratio": 0.5}
    return {
        "n_estimators": 8,
        "max_depth": 3,
        "max_samples": 1.0,
        "random_state": 0,
        "n_jobs": 1,
    }


@pytest.mark.parametrize(("model_name", "entrypoint"), _PARITY_CASES)
def test_construct_model_matches_from_resolved_config(model_name: str, entrypoint: str) -> None:
    hp = _minimal_hp(model_name)
    model_cls = _model_class(model_name, entrypoint)
    if model_name.startswith("Naive"):
        mudataset = synthetic_mudataset_identity()
    else:
        mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()

    config = (
        from_spec("pca[expression]:identity:randomForest", hyperparameters=hp)
        if entrypoint == "construct_model"
        else from_spec(model_name, hyperparameters=hp)
    )
    flat_hp = model_cls.get_default_hyperparameters() if not hp else hp
    facade = model_cls(flat_hp)
    facade.train(mudataset, split)
    facade_preds = facade.predict(mudataset, split)

    direct = model_cls._from_resolved_config(config)
    direct.train(mudataset, split)
    direct_preds = direct.predict(mudataset, split)

    assert facade._resolved_model_config is not None
    assert facade._resolved_model_config.predictor_name == (
        config.predictor_name if hasattr(config, "predictor_name") else config.predictor.name
    )
    assert np.allclose(facade_preds, direct_preds, equal_nan=True)


@pytest.mark.parametrize(("model_name", "entrypoint"), _PARITY_CASES)
def test_construct_model_save_load_preserves_predictions(model_name: str, entrypoint: str) -> None:
    hp = _minimal_hp(model_name)
    model_cls = _model_class(model_name, entrypoint)
    flat_hp = model_cls.get_default_hyperparameters() if not hp else hp
    if model_name.startswith("Naive"):
        mudataset = synthetic_mudataset_identity()
    else:
        mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()

    model = model_cls(flat_hp)
    model.train(mudataset, split)
    before_preds = model.predict(mudataset, split)
    assert model._stack is not None
    before_state = model._stack.component_state()

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = f"{model_dir}/model"
        model.save(checkpoint)
        loaded = model_cls.load(checkpoint)
        after_preds = loaded.predict(mudataset, split)
        assert loaded._stack is not None
        after_state = loaded._stack.component_state()

    assert np.allclose(before_preds, after_preds, equal_nan=True)
    assert before_state.keys() == after_state.keys()
