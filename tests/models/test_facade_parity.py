"""Facade vs direct ModelConfig.create_model() parity."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models.config import ModelConfig
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    identity_cell_line_features,
    identity_drug_features,
    multi_drug_response,
)

_PARITY_MODELS = ("NaivePredictor", "NaiveDrugMeanPredictor", "ElasticNet", "RandomForest")


@pytest.mark.parametrize("model_name", _PARITY_MODELS)
def test_facade_matches_direct_composed_model(model_name: str) -> None:
    response = multi_drug_response()
    if model_name.startswith("Naive"):
        cell_line_input = identity_cell_line_features()
        drug_input = identity_drug_features()
        hp: dict = {}
    else:
        cell_line_input = cell_line_gene_expression()
        drug_input = drug_fingerprints()
        hp = (
            {"alpha": 0.1, "l1_ratio": 0.5}
            if model_name == "ElasticNet"
            else {
                "n_estimators": 8,
                "max_depth": 3,
                "max_samples": 1.0,
                "random_state": 0,
                "n_jobs": 1,
            }
        )

    config = ModelConfig.from_spec(model_name, hyperparameters=hp)
    facade = MODEL_FACTORY[model_name]()
    facade.build_from_model_config(config)
    facade.train(response, cell_line_input, drug_input)
    facade_preds = facade.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)

    direct = config.create_model()
    direct.train(response, cell_line_input, drug_input)
    direct_preds = direct.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)

    assert facade._resolved_model_config is not None
    assert facade._resolved_model_config.predictor.name == config.predictor.name
    assert np.allclose(facade_preds, direct_preds, equal_nan=True)
