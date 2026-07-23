"""ComposedModel execution through ModelConfig."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import ModelConfig


def test_sklearn_model_config_builds_composed_model() -> None:
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    model = config.create_model()
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
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)


def test_ridge_predictor_via_recipe() -> None:
    model = ModelConfig.from_spec("scaledGeneExpression:fingerprints:ridge", hyperparameters={"alpha": 1.0}).create_model()
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2])},
            "cl2": {"gene_expression": np.array([0.3, 0.4])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0])},
            "d2": {"fingerprints": np.array([0.0])},
        }
    )
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (2,)


def test_naive_model_config_train_predict_on_synthetic_data() -> None:
    model = ModelConfig.from_spec("NaivePredictor").create_model()
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    empty = FeatureDataset(features={})
    model.train(response, empty, empty)
    preds = model.predict(response.cell_line_ids, response.drug_ids, empty, empty)
    assert np.allclose(preds, np.array([2.0, 2.0]))
