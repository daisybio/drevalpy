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


def _fitted_stack(
    sample_input: MockFeatureSource, *, n_jobs: int | None = None
) -> tuple[_ModelPipeline, SingleDrugSklearnPredictor]:
    config = from_spec(
        "scaledGeneExpression:identity:singleDrugElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5}
    )
    stack = _ModelPipeline(config)
    stack.reset()
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 10.0, 12.0]) if n_jobs else np.array([1.0, 1.0, 10.0, 10.0]),
        sample_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
    )
    context = TrainingContext(n_jobs=n_jobs) if n_jobs else None
    stack._fit_featurizers_and_predictor(response, sample_input, training_context=context)
    predictor = stack._predictor
    assert isinstance(predictor, SingleDrugSklearnPredictor)
    return stack, predictor


def _fitted_predictor() -> SingleDrugSklearnPredictor:
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


def test_state_round_trips_every_per_drug_estimator_and_shares_entries_with_the_base() -> None:
    predictor = _fitted_predictor()
    state = predictor.get_state()
    assert set(state) == {"estimators", "hyperparameters", "mode"}
    assert SingleDrugElasticNetPredictor().is_fitted() is False

    restored = SingleDrugElasticNetPredictor()
    restored.set_state(state)

    assert restored.is_fitted() is True
    assert set(restored._estimators) == {"d1", "d2"}
    assert restored._h["alpha"] == pytest.approx(0.1)
    assert restored._mode is PredictionMode.REGRESSION


@pytest.mark.parametrize(
    ("entry", "value", "message"),
    [
        ("estimators", {}, "missing fitted per-drug estimators"),
        ("hyperparameters", {}, "missing hyperparameters"),
        ("mode", 3, "invalid prediction mode"),
    ],
)
def test_malformed_state_is_rejected_and_leaves_the_estimators_untouched(
    entry: str, value: object, message: str
) -> None:
    predictor = _fitted_predictor()
    state = predictor.get_state()
    state[entry] = value

    with pytest.raises(PredictorStateError, match=message):
        predictor.set_state(state)

    assert set(predictor._estimators) == {"d1", "d2"}
    with pytest.raises(PredictorStateError, match="missing fitted per-drug estimators"):
        SingleDrugElasticNetPredictor().set_state({"estimators": {}, "hyperparameters": {}})


def test_n_jobs_does_not_change_the_estimators_and_the_parallel_state_restores() -> None:
    sample_input = _sample_input()
    serial = _fitted_stack(sample_input, n_jobs=1)[1]
    parallel = _fitted_stack(sample_input, n_jobs=2)[1]

    assert list(serial._estimators) == list(parallel._estimators) == ["d1", "d2"]
    for drug_id, estimator in serial._estimators.items():
        np.testing.assert_array_equal(estimator.coef_, parallel._estimators[drug_id].coef_)
        assert estimator.intercept_ == parallel._estimators[drug_id].intercept_
    restored = SingleDrugElasticNetPredictor()
    restored.set_state(parallel.get_state())
    assert restored.is_fitted() is True
    assert set(restored._estimators) == {"d1", "d2"}
