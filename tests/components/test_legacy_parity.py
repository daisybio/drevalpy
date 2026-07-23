"""Facade/component parity for simple models (formerly legacy parity)."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.config import ModelConfig


def _synthetic_sklearn_data() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )
    return response, cell_line_input, drug_input


def test_elastic_net_facade_matches_model_config() -> None:
    response, cell_line_input, drug_input = _synthetic_sklearn_data()
    hp = {"alpha": 0.1, "l1_ratio": 0.5}
    facade = MODEL_FACTORY["ElasticNet"]()
    facade.build_model(hp)
    facade.train(response, cell_line_input, drug_input)
    facade_preds = facade.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)

    direct = ModelConfig.from_spec("ElasticNet", hyperparameters=hp).create_model()
    direct.train(response, cell_line_input, drug_input)
    direct_preds = direct.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(facade_preds, direct_preds)


def test_naive_predictor_facade_matches_model_config() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0, 5.0]),
        cell_line_ids=np.array(["cl1", "cl2", "cl3"]),
        drug_ids=np.array(["d1", "d2", "d3"]),
    )
    empty = FeatureDataset(features={})
    facade = MODEL_FACTORY["NaivePredictor"]()
    facade.build_model({})
    facade.train(response, empty, empty)
    facade_preds = facade.predict(response.cell_line_ids, response.drug_ids, empty, empty)

    direct = ModelConfig.from_spec("NaivePredictor").create_model()
    direct.train(response, empty, empty)
    direct_preds = direct.predict(response.cell_line_ids, response.drug_ids, empty, empty)
    assert np.allclose(facade_preds, direct_preds)


def test_naive_drug_mean_facade_matches_model_config() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0, 5.0, 7.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    empty = FeatureDataset(features={})
    facade = MODEL_FACTORY["NaiveDrugMeanPredictor"]()
    facade.build_model({})
    facade.train(response, empty, empty)
    facade_preds = facade.predict(response.cell_line_ids, response.drug_ids, empty, empty)

    direct = ModelConfig.from_spec("NaiveDrugMeanPredictor").create_model()
    direct.train(response, empty, empty)
    direct_preds = direct.predict(response.cell_line_ids, response.drug_ids, empty, empty)
    assert np.allclose(facade_preds, direct_preds)


def test_model_factory_names_still_instantiate() -> None:
    for name in ("ElasticNet", "NaivePredictor", "RandomForest", "SingleDrugElasticNet"):
        model = MODEL_FACTORY[name]()
        assert model.get_model_name() == name
