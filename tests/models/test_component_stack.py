"""Component stack execution through construct_model and build_component_stack."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import construct_model
from drevalpy.models._component_stack import build_component_stack
from drevalpy.models.config import ModelConfig


def test_sklearn_model_config_builds_runnable_model() -> None:
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    model = construct_model("ElasticNet", config)()
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
    assert np.isfinite(preds).all()


def test_build_component_stack_train_predict() -> None:
    config = ModelConfig.from_spec("scaledGeneExpression:fingerprints:ridge", hyperparameters={"alpha": 1.0})
    stack = build_component_stack(config)
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
    stack.train(response, cell_line_input, drug_input)
    preds = stack.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (2,)


def test_naive_model_train_predict_on_synthetic_data() -> None:
    model = construct_model("NaivePredictor")()
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    empty = FeatureDataset(features={})
    model.train(response, empty, empty)
    preds = model.predict(response.cell_line_ids, response.drug_ids, empty, empty)
    assert np.allclose(preds, np.array([2.0, 2.0]))


def test_untrained_model_predict_raises() -> None:
    model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
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
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)


def test_model_has_no_predictor_hyperparameter_mutator() -> None:
    model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
    assert not hasattr(model, "update_predictor_hyperparameters")
    assert model._resolved_model_config is not None
    assert model._resolved_model_config.predictor.hyperparameters["alpha"] == 0.1


def test_druggnn_stack_configures_both_featurizers() -> None:
    stack = build_component_stack(ModelConfig.from_spec("DrugGNN"))
    assert stack._cell_line_featurizer is not None
    assert stack._drug_featurizer is not None
