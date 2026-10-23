"""Tests for per-drug sklearn predictor routing."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_models import SingleDrugElasticNetPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models._pipeline import _ModelPipeline
from drevalpy.models.config import from_spec
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.enums.prediction_mode import PredictionMode
from tests.conftest import MockFeatureSource


def _sample_input() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )


def _fitted_stack(sample_input: MockFeatureSource) -> tuple[object, SingleDrugSklearnPredictor]:
    """Fit a per-drug elastic-net stack on two drugs over three genes.

    :param sample_input: Feature source the stack fits its featurizers on.
    :returns: ``(stack, predictor)``, with the predictor narrowed to the per-drug type.
    """
    config = from_spec(
        "scaledGeneExpression:identity:singleDrugElasticNet",
        hyperparameters={"alpha": 0.1, "l1_ratio": 0.5},
    )
    stack = _ModelPipeline(config)
    stack.reset()
    response = ResponseBatch(
        response=np.array([1.0, 1.0, 10.0, 10.0]),
        sample_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
    )
    stack._fit_featurizers_and_predictor(response, sample_input)
    predictor = stack._predictor
    assert isinstance(predictor, SingleDrugSklearnPredictor)
    return stack, predictor


def _fitted_predictor() -> SingleDrugSklearnPredictor:
    """Return a per-drug predictor fitted on two drugs over three genes.

    :returns: The fitted predictor.
    """
    return _fitted_stack(_sample_input())[1]


def test_identity_routes_estimators_without_entering_design_matrix() -> None:
    sample_input = _sample_input()

    stack, predictor = _fitted_stack(sample_input)

    assert set(predictor._estimators) == {"d1", "d2"}
    assert {estimator.n_features_in_ for estimator in predictor._estimators.values()} == {3}

    predictions = predictor.predict(
        stack._batch(
            ResponseBatch(response=None, sample_ids=np.array(["cl1", "cl1"]), drug_ids=np.array(["d1", "d2"])),
            sample_input,
            None,
        )
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


class TestNJobs:
    """Fitting the drugs in worker processes gives the estimators of fitting them one after the other."""

    @staticmethod
    def _fit(n_jobs: int) -> SingleDrugSklearnPredictor:
        sample_input = _sample_input()
        stack = _ModelPipeline(
            from_spec(
                "scaledGeneExpression:identity:singleDrugElasticNet",
                hyperparameters={"alpha": 0.1, "l1_ratio": 0.5},
            )
        )
        stack.reset()
        response = ResponseBatch(
            response=np.array([1.0, 2.0, 10.0, 12.0]),
            sample_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
            drug_ids=np.array(["d1", "d1", "d2", "d2"]),
        )
        stack._fit_featurizers_and_predictor(response, sample_input, training_context=TrainingContext(n_jobs=n_jobs))
        predictor = stack._predictor
        assert isinstance(predictor, SingleDrugSklearnPredictor)
        return predictor

    def test_the_estimators_do_not_depend_on_it(self) -> None:
        serial, parallel = self._fit(1), self._fit(2)

        assert list(serial._estimators) == list(parallel._estimators) == ["d1", "d2"]
        for drug_id, estimator in serial._estimators.items():
            np.testing.assert_array_equal(estimator.coef_, parallel._estimators[drug_id].coef_)
            assert estimator.intercept_ == parallel._estimators[drug_id].intercept_

    def test_the_state_of_a_parallel_fit_restores_like_a_serial_one(self) -> None:
        restored = SingleDrugElasticNetPredictor()
        restored.set_state(self._fit(2).get_state())

        assert restored.is_fitted() is True
        assert set(restored._estimators) == {"d1", "d2"}
