"""Tests for the shared single-drug routing helpers and the per-drug sklearn predictors built on them."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from drevalpy._utils.seed import seed_everything
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    routing_keys,
    run_per_drug,
)
from drevalpy.components.predictors.sklearn_models import SingleDrugElasticNetPredictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors._helpers import pickle_round_trip


def _routing_batch() -> ModelInputBatch:
    return ModelInputBatch.from_response(
        ResponseBatch(
            response=np.array([1.0, 2.0, 3.0, 4.0]),
            sample_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
            drug_ids=np.array(["d1", "d1", "d2", "d2"]),
        ),
        sample=FeaturizerResult.from_blocks({}, np.array(["cl1", "cl2"])),
    )


def test_routing_keys_are_the_drug_ids_and_masks_are_stable() -> None:
    batch = _routing_batch()
    assert batch.drug is None
    assert routing_keys(batch).tolist() == ["d1", "d1", "d2", "d2"]
    routed = list(iter_drug_masks(batch))
    assert {drug_id for drug_id, _ in routed} == {"d1", "d2"}
    for drug_id, mask in routed:
        assert routing_keys(batch.subset_pairs(mask)).tolist() == [drug_id] * int(mask.sum())


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


def _batch(response: ResponseBatch, *, n_jobs: int | None = None) -> ModelInputBatch:
    sample = FeaturizerResult.from_blocks(
        {"gene_expression": numeric_feature_block(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))},
        np.array(["cl1", "cl2"]),
    )
    context = TrainingContext(n_jobs=n_jobs) if n_jobs else None
    return ModelInputBatch.from_response(response, sample=sample, training_context=context)


def _fitted_predictor(*, n_jobs: int | None = None) -> SingleDrugElasticNetPredictor:
    predictor = SingleDrugElasticNetPredictor({"alpha": 0.1, "l1_ratio": 0.5})
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 10.0, 12.0]) if n_jobs else np.array([1.0, 1.0, 10.0, 10.0]),
        sample_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
    )
    predictor.fit(_batch(response, n_jobs=n_jobs))
    return predictor


def test_drug_ids_route_estimators_without_entering_design_matrix() -> None:
    predictor = _fitted_predictor()

    assert SingleDrugElasticNetPredictor.drug_inputs.kind == "none"
    assert set(predictor._estimators) == {"d1", "d2"}
    assert {estimator.n_features_in_ for estimator in predictor._estimators.values()} == {3}

    predictions = predictor.predict(
        _batch(
            ResponseBatch(
                response=None, sample_ids=np.array(["cl1", "cl1", "cl1"]), drug_ids=np.array(["d1", "d2", "d3"])
            )
        )
    )
    assert np.allclose(predictions[:2], np.array([1.0, 10.0]))
    assert np.isnan(predictions[2])


def test_pickle_round_trips_every_per_drug_estimator() -> None:
    predictor = _fitted_predictor()
    assert SingleDrugElasticNetPredictor().is_fitted() is False

    restored = pickle_round_trip(predictor)

    assert restored.is_fitted() is True
    assert set(restored._estimators) == {"d1", "d2"}
    assert restored._hyperparameters["alpha"] == pytest.approx(0.1)


def test_n_jobs_does_not_change_the_estimators() -> None:
    serial = _fitted_predictor(n_jobs=1)
    parallel = _fitted_predictor(n_jobs=2)

    assert list(serial._estimators) == list(parallel._estimators) == ["d1", "d2"]
    for drug_id, estimator in serial._estimators.items():
        np.testing.assert_array_equal(estimator.coef_, parallel._estimators[drug_id].coef_)
        assert estimator.intercept_ == parallel._estimators[drug_id].intercept_
    restored = pickle_round_trip(parallel)
    assert set(restored._estimators) == {"d1", "d2"}
