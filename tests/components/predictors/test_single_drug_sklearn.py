"""Tests for per-drug sklearn predictor routing."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_models import SingleDrugElasticNetPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models.component_stack import build_component_stack
from drevalpy.models.config import from_spec
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.enums.prediction_mode import PredictionMode
from tests.conftest import MockFeatureSource


def _cell_line_input() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )


def _fitted_stack(cell_line_input: MockFeatureSource) -> tuple[object, SingleDrugSklearnPredictor]:
    """Fit a per-drug elastic-net stack on two drugs over three genes.

    :param cell_line_input: Feature source the stack fits its featurizers on.
    :returns: ``(stack, predictor)``, with the predictor narrowed to the per-drug type.
    """
    config = from_spec(
        "scaledGeneExpression:identity:singleDrugElasticNet",
        hyperparameters={"alpha": 0.1, "l1_ratio": 0.5},
    )
    stack = build_component_stack(config)
    response = ResponseBatch(
        response=np.array([1.0, 1.0, 10.0, 10.0]),
        cell_line_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
    )
    stack._fit_featurizers_and_predictor(response, cell_line_input)
    predictor = stack._predictor
    assert isinstance(predictor, SingleDrugSklearnPredictor)
    return stack, predictor


def _fitted_predictor() -> SingleDrugSklearnPredictor:
    """Return a per-drug predictor fitted on two drugs over three genes.

    :returns: The fitted predictor.
    """
    return _fitted_stack(_cell_line_input())[1]


def test_identity_routes_estimators_without_entering_design_matrix() -> None:
    cell_line_input = _cell_line_input()

    stack, predictor = _fitted_stack(cell_line_input)

    assert set(predictor._estimators) == {"d1", "d2"}
    assert {estimator.n_features_in_ for estimator in predictor._estimators.values()} == {3}

    predictions = stack.predict_from_features(
        np.array(["cl1", "cl1"]),
        np.array(["d1", "d2"]),
        cell_line_input,
    )
    assert np.allclose(predictions, np.array([1.0, 10.0]))


class TestState:
    """The per-drug predictor swaps the estimator entry but shares the rest with its base."""

    def test_state_carries_the_estimators_plus_the_shared_entries(self):
        assert set(_fitted_predictor().get_state()) == {"estimators", "hyperparameters", "mode"}

    def test_is_not_fitted_without_estimators(self):
        assert SingleDrugElasticNetPredictor().is_fitted() is False

    def test_round_trip_restores_every_per_drug_estimator(self):
        predictor = _fitted_predictor()

        restored = SingleDrugElasticNetPredictor()
        restored.set_state(predictor.get_state())

        assert restored.is_fitted() is True
        assert set(restored._estimators) == {"d1", "d2"}
        assert restored._h["alpha"] == pytest.approx(0.1)
        assert restored._mode is PredictionMode.REGRESSION

    def test_rejects_a_state_without_estimators(self):
        state = _fitted_predictor().get_state()
        state["estimators"] = {}

        with pytest.raises(PredictorStateError, match="missing fitted per-drug estimators"):
            SingleDrugElasticNetPredictor().set_state(state)

    def test_the_missing_estimators_error_takes_precedence_over_the_shared_ones(self):
        with pytest.raises(PredictorStateError, match="missing fitted per-drug estimators"):
            SingleDrugElasticNetPredictor().set_state({"estimators": {}, "hyperparameters": {}})

    def test_rejects_a_state_without_hyperparameters(self):
        state = _fitted_predictor().get_state()
        state["hyperparameters"] = {}

        with pytest.raises(PredictorStateError, match="missing hyperparameters"):
            SingleDrugElasticNetPredictor().set_state(state)

    def test_rejects_an_unusable_prediction_mode(self):
        state = _fitted_predictor().get_state()
        state["mode"] = 3

        with pytest.raises(PredictorStateError, match="invalid prediction mode"):
            SingleDrugElasticNetPredictor().set_state(state)

    def test_a_rejected_state_leaves_the_estimators_untouched(self):
        predictor = _fitted_predictor()
        state = predictor.get_state()
        state["mode"] = 3

        with pytest.raises(PredictorStateError, match="invalid prediction mode"):
            predictor.set_state(state)

        assert set(predictor._estimators) == {"d1", "d2"}
