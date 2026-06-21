"""Parity tests between legacy MODEL_FACTORY models and modular configs."""

from __future__ import annotations

import numpy as np

from drevalpy.components.factory import model_config_for_name, sklearn_model_config
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import MODEL_FACTORY


def _synthetic_data() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0, 2.5, 3.5]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3, 0.4])},
            "cl2": {"gene_expression": np.array([0.5, 0.6, 0.7, 0.8])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0, 0.5, 0.2])},
            "d2": {"fingerprints": np.array([0.0, 1.0, 0.3, 0.7])},
        }
    )
    return response, cell_line_input, drug_input


def test_elastic_net_legacy_matches_model_config() -> None:
    register_builtin_components()
    response, cell_line_input, drug_input = _synthetic_data()
    hyperparameters = {"alpha": 0.05, "l1_ratio": 0.4, "max_iter": 5000}

    legacy = MODEL_FACTORY["ElasticNet"]()
    legacy.build_model(hyperparameters=hyperparameters)
    legacy.train(response, cell_line_input, drug_input)
    legacy_preds = legacy.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )

    config = model_config_for_name("ElasticNet", hyperparameters)
    composed = config.create_model()
    composed.train(response, cell_line_input, drug_input)
    composed_preds = composed.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )

    assert np.allclose(legacy_preds, composed_preds, rtol=1e-6, atol=1e-6)


def test_factory_and_zoo_elastic_net_configs_align() -> None:
    factory_config = sklearn_model_config("elasticNet", {"alpha": 0.1})
    zoo_config = model_config_for_name("ElasticNet", {"alpha": 0.1})
    assert factory_config.cell_line_featurizer.type == zoo_config.cell_line_featurizer.type
    assert factory_config.drug_featurizer.type == zoo_config.drug_featurizer.type
    assert factory_config.predictor.type == zoo_config.predictor.type
    assert zoo_config.predictor.hyperparameters["alpha"] == 0.1
