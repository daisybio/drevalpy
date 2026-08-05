"""Tests for per-drug sklearn predictor routing."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_stack import build_component_stack
from drevalpy.models.config import ModelConfig


def test_identity_routes_estimators_without_entering_design_matrix() -> None:
    config = ModelConfig.from_spec(
        "scaledGeneExpression:identity:singleDrugElasticNet",
        hyperparameters={"alpha": 0.1, "l1_ratio": 0.5},
    )
    stack = build_component_stack(config)
    response = DrugResponseDataset(
        response=np.array([1.0, 1.0, 10.0, 10.0]),
        cell_line_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )

    stack.train(response, cell_line_input)

    predictor = stack._predictor
    assert isinstance(predictor, SingleDrugSklearnPredictor)
    assert set(predictor._estimators) == {"d1", "d2"}
    assert {estimator.n_features_in_ for estimator in predictor._estimators.values()} == {3}

    predictions = stack.predict(
        np.array(["cl1", "cl1"]),
        np.array(["d1", "d2"]),
        cell_line_input,
    )
    assert np.allclose(predictions, np.array([1.0, 10.0]))
