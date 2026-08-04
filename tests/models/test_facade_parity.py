"""construct_model vs ModelConfig._from_resolved_config parity."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)

_PARITY_CASES = (
    ("NaivePredictor", "factory"),
    ("NaiveDrugMeanPredictor", "factory"),
    ("ElasticNet", "factory"),
    ("RandomForest", "factory"),
    ("PcaIdentityRF", "construct_model"),
)


def _training_inputs(model_name: str) -> tuple:
    response = multi_drug_response()
    if model_name.startswith("Naive"):
        return response, identity_cell_line_features(), identity_drug_features(), {}
    if model_name == "ElasticNet":
        hp = {"alpha": 0.1, "l1_ratio": 0.5}
    else:
        hp = {
            "n_estimators": 8,
            "max_depth": 3,
            "max_samples": 1.0,
            "random_state": 0,
            "n_jobs": 1,
        }
    return response, cell_line_gene_expression(), drug_fingerprints(), hp


def _model_class(model_name: str, entrypoint: str):
    if entrypoint == "construct_model":
        return construct_model(model_name, "pca[expression]:identity:randomForest")
    return construct_model(model_name)


@pytest.mark.parametrize(("model_name", "entrypoint"), _PARITY_CASES)
def test_construct_model_matches_from_resolved_config(model_name: str, entrypoint: str) -> None:
    response, cell_line_input, drug_input, hp = _training_inputs(model_name)
    model_cls = _model_class(model_name, entrypoint)

    config = (
        ModelConfig.from_spec("pca[expression]:identity:randomForest", hyperparameters=hp)
        if entrypoint == "construct_model"
        else ModelConfig.from_spec(model_name, hyperparameters=hp)
    )
    flat_hp = model_cls.get_default_hyperparameters() if not hp else hp
    facade = model_cls(flat_hp)
    facade.train(response, cell_line_input, drug_input)
    facade_preds = facade.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)

    direct = model_cls._from_resolved_config(config)
    direct.train(response, cell_line_input, drug_input)
    direct_preds = direct.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)

    assert facade._resolved_model_config is not None
    assert facade._resolved_model_config.predictor.name == config.predictor.name
    assert np.allclose(facade_preds, direct_preds, equal_nan=True)


@pytest.mark.parametrize(("model_name", "entrypoint"), _PARITY_CASES)
def test_construct_model_save_load_preserves_predictions(model_name: str, entrypoint: str) -> None:
    response, cell_line_input, drug_input, hp = _training_inputs(model_name)
    model_cls = _model_class(model_name, entrypoint)
    flat_hp = model_cls.get_default_hyperparameters() if not hp else hp

    model = model_cls(flat_hp)
    model.train(response, cell_line_input, drug_input)
    before_preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert model._stack is not None
    before_state = model._stack.component_state()

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = f"{model_dir}/model"
        model.save(checkpoint)
        loaded = model_cls.load(checkpoint)
        after_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
        assert loaded._stack is not None
        after_state = loaded._stack.component_state()

    assert np.allclose(before_preds, after_preds, equal_nan=True)
    assert before_state.keys() == after_state.keys()
