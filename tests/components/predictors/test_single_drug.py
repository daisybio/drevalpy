"""Tests for the shared single-drug routing helpers and the per-drug sklearn predictors built on them."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from drevalpy._utils.seed import seed_everything
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
    run_per_drug,
)
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_models import SingleDrugElasticNetPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models._pipeline import _ModelPipeline
from drevalpy.models.config import from_spec
from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.enums.prediction_mode import PredictionMode
from tests.conftest import MockFeatureSource


def _identity_batch() -> ModelInputBatch:
    return ModelInputBatch(
        sample_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        sample=FeaturizerResult(np.array(["cl1", "cl2"]), {}, ()),
        drug=FeaturizerResult(
            np.array(["d1", "d2"]),
            {
                "identity": FeatureBlock(
                    values=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    format=FeatureFormat.NUMERIC_MATRIX,
                ),
                "identity_categories": FeatureBlock(values=np.array(["d1", "d2"]), format=FeatureFormat.NUMERIC_MATRIX),
            },
            ("identity",),
        ),
        sample_pair_idx=np.array([0, 1, 0, 1]),
        drug_pair_idx=np.array([0, 0, 1, 1]),
    )


def test_routing_keys_decode_known_drugs_and_masks_are_stable() -> None:
    batch = _identity_batch()
    assert routing_keys(batch).tolist() == ["d1", "d1", "d2", "d2"]
    routed = list(iter_drug_masks(batch))
    assert {drug_id for drug_id, _ in routed} == {"d1", "d2"}
    for drug_id, mask in routed:
        assert routing_keys(batch.subset_pairs(mask)).tolist() == [drug_id] * int(mask.sum())


def test_routing_rejects_unknown_keys_and_missing_identity_blocks() -> None:
    with pytest.raises(ValueError, match="unknown drug identities"):
        require_known_training_keys(np.array(["d1", ""]))
    batch = _identity_batch()
    batch.drug = FeaturizerResult(np.array(["d1", "d2"]), {}, ())
    with pytest.raises(ValueError, match="require drug identity features"):
        routing_keys(batch)


def _draw(drug_id: str, scale: float) -> tuple[float, float, float, int, int]:
    import torch

    return (
        scale * random.getrandbits(24),
        scale * float(np.random.rand()),
        scale * float(torch.rand(1)),
        torch.get_num_threads(),
        os.getpid(),
    )


def _fail_on_d2(drug_id: str, task: int) -> int:
    if drug_id == "d2":
        raise RuntimeError("d2 does not converge")
    return task


def _tasks(*drugs: str) -> list[tuple[str, float]]:
    return [(drug, 1.0) for drug in drugs]


@pytest.fixture(autouse=True)
def _restore_global_rngs():
    import torch

    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])


def test_run_per_drug_returns_models_in_task_order_and_nothing_for_no_drugs() -> None:
    fitted = run_per_drug(lambda drug_id, task: (drug_id, task), [("d2", 2), ("d1", 1), ("d3", 3)])

    assert list(fitted) == ["d2", "d1", "d3"]
    assert fitted == {"d2": ("d2", 2), "d1": ("d1", 1), "d3": ("d3", 3)}
    assert run_per_drug(_draw, [], n_jobs=2) == {}


def test_n_jobs_changes_neither_the_draws_nor_the_final_rng_state() -> None:
    results, after = {}, {}
    for n_jobs in (1, 2):
        seed_everything(11)
        results[n_jobs] = run_per_drug(_draw, _tasks("d1", "d2", "d3"), n_jobs=n_jobs)
        after[n_jobs] = random.getrandbits(24), float(np.random.rand())

    in_process = {drug: draws[:3] for drug, draws in results[1].items()}
    assert in_process == {drug: draws[:3] for drug, draws in results[2].items()}
    assert len({draws for draws in in_process.values()}) == 3
    assert after[1] == after[2]


def test_a_drug_depends_on_the_global_seed_but_not_on_the_other_drugs() -> None:
    seed_everything(11)
    alone = run_per_drug(_draw, _tasks("d1", "d2"))
    seed_everything(11)
    with_d3 = run_per_drug(_draw, _tasks("d3", "d2", "d1"))
    seed_everything(12)
    reseeded = run_per_drug(_draw, _tasks("d1", "d2"))

    assert alone["d2"][:3] == with_d3["d2"][:3]
    assert alone["d1"][:3] != reseeded["d1"][:3]


def test_a_lone_drug_runs_in_the_calling_process_and_workers_are_single_threaded_processes() -> None:
    seed_everything(5)
    expected = random.getrandbits(24)
    seed_everything(5)
    (draws,) = run_per_drug(_draw, _tasks("d1"), n_jobs=2).values()
    assert draws[0] == expected
    assert draws[4] == os.getpid()

    workers = run_per_drug(_draw, _tasks("d1", "d2"), n_jobs=2)
    assert {draws[4] for draws in workers.values()}.isdisjoint({os.getpid()})
    assert {draws[3] for draws in workers.values()} == {1}
    serial = run_per_drug(_draw, _tasks("d1", "d2"), n_jobs=1)
    assert {draws[4] for draws in serial.values()} == {os.getpid()}


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_an_error_for_one_drug_is_raised_for_all(n_jobs: int) -> None:
    with pytest.raises(RuntimeError, match="d2 does not converge"):
        run_per_drug(_fail_on_d2, [("d1", 1), ("d2", 2), ("d3", 3)], n_jobs=n_jobs)


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
