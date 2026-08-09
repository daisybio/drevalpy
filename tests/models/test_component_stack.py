"""Component stack execution through construct_model and build_component_stack."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data.structures.response_batch import ResponseBatch
from drevalpy.models import construct_model
from drevalpy.models._component_stack import build_component_stack
from drevalpy.models.config import from_spec
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)


def test_sklearn_model_config_builds_runnable_model() -> None:
    model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape == (2,)
    assert np.isfinite(preds).all()


def test_build_component_stack_train_predict() -> None:
    config = from_spec("scaledGeneExpression:fingerprints:ridge", hyperparameters={"alpha": 1.0})
    stack = build_component_stack(config)
    response = ResponseBatch(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    cell_line_input = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2])},
            "cl2": {"gene_expression": np.array([0.3, 0.4])},
        }
    )
    drug_input = MockFeatureSource(
        features={
            "d1": {"fingerprints": np.array([1.0])},
            "d2": {"fingerprints": np.array([0.0])},
        }
    )
    stack._fit_featurizers_and_predictor(response, cell_line_input, drug_input)
    preds = stack.predict_from_features(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (2,)


def test_naive_model_train_predict_on_synthetic_data() -> None:
    model = construct_model("NaivePredictor")()
    mudataset = synthetic_mudataset_identity()
    split = lco_split_masks()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert np.isfinite(preds).all()


def test_untrained_model_predict_raises() -> None:
    model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lco_split_masks()
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(mudataset, split)


def test_model_has_no_predictor_hyperparameter_mutator() -> None:
    model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
    assert not hasattr(model, "update_predictor_hyperparameters")
    assert model._resolved_model_config is not None
    assert model._resolved_model_config.predictor_values()["alpha"] == 0.1


def test_druggnn_stack_configures_both_featurizers() -> None:
    stack = build_component_stack(from_spec("DrugGNN"))
    assert stack._cell_line_featurizer is not None
    assert stack._drug_featurizer is not None
