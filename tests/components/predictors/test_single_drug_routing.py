"""Tests for shared single-drug routing helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
    run_per_drug,
)
from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.utils.seed import seed_everything


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


def test_routing_keys_decode_known_drugs() -> None:
    keys = routing_keys(_identity_batch())
    assert keys.tolist() == ["d1", "d1", "d2", "d2"]


def test_iter_drug_masks_returns_stable_masks() -> None:
    batch = _identity_batch()
    routed = list(iter_drug_masks(batch))
    assert {drug_id for drug_id, _ in routed} == {"d1", "d2"}
    for drug_id, mask in routed:
        assert routing_keys(batch.subset_pairs(mask)).tolist() == [drug_id] * int(mask.sum())


def test_require_known_training_keys_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown drug identities"):
        require_known_training_keys(np.array(["d1", ""]))


def test_routing_keys_requires_identity_blocks() -> None:
    batch = _identity_batch()
    batch.drug = FeaturizerResult(np.array(["d1", "d2"]), {}, ())
    with pytest.raises(ValueError, match="require drug identity features"):
        routing_keys(batch)


def _draw(drug_id: str, scale: float) -> tuple[float, float, float, int, int]:
    """Stand-in for a fit: draws from every RNG the predictors use, and reports where it ran."""
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


class TestRunPerDrug:
    """``run_per_drug`` is the one place a per-drug fit fans out, whatever the predictor."""

    def test_returns_the_fitted_model_per_drug_in_task_order(self) -> None:
        fitted = run_per_drug(lambda drug_id, task: (drug_id, task), [("d2", 2), ("d1", 1), ("d3", 3)])

        assert list(fitted) == ["d2", "d1", "d3"]
        assert fitted == {"d2": ("d2", 2), "d1": ("d1", 1), "d3": ("d3", 3)}

    def test_no_drugs_fit_nothing(self) -> None:
        assert run_per_drug(_draw, [], n_jobs=2) == {}

    def test_n_jobs_does_not_change_what_is_fitted(self) -> None:
        results = {}
        for n_jobs in (1, 2):
            seed_everything(11)
            results[n_jobs] = run_per_drug(_draw, _tasks("d1", "d2", "d3"), n_jobs=n_jobs)

        in_process = {drug: draws[:3] for drug, draws in results[1].items()}
        in_workers = {drug: draws[:3] for drug, draws in results[2].items()}
        assert in_process == in_workers
        assert len({draws[:3] for draws in in_process.values()}) == 3

    def test_the_global_rngs_end_in_the_same_state_for_every_n_jobs(self) -> None:
        after = {}
        for n_jobs in (1, 2):
            seed_everything(11)
            run_per_drug(_draw, _tasks("d1", "d2"), n_jobs=n_jobs)
            after[n_jobs] = random.getrandbits(24), float(np.random.rand())

        assert after[1] == after[2]

    def test_a_drug_does_not_depend_on_the_other_drugs(self) -> None:
        seed_everything(11)
        alone_with_d1 = run_per_drug(_draw, _tasks("d1", "d2"))["d2"][:3]
        seed_everything(11)
        with_d3 = run_per_drug(_draw, _tasks("d3", "d2", "d1"))["d2"][:3]

        assert alone_with_d1 == with_d3

    def test_the_global_seed_decides_the_draws(self) -> None:
        seed_everything(1)
        first = run_per_drug(_draw, _tasks("d1", "d2"))
        seed_everything(2)
        second = run_per_drug(_draw, _tasks("d1", "d2"))

        assert first["d1"][:3] != second["d1"][:3]

    def test_a_lone_drug_runs_in_the_calling_process_with_the_rng_state_it_finds(self) -> None:
        seed_everything(5)
        expected = random.getrandbits(24)
        seed_everything(5)

        (draws,) = run_per_drug(_draw, _tasks("d1"), n_jobs=2).values()

        assert draws[0] == expected
        assert draws[4] == os.getpid()

    def test_workers_are_other_processes_with_one_thread(self) -> None:
        fitted = run_per_drug(_draw, _tasks("d1", "d2"), n_jobs=2)

        assert {draws[4] for draws in fitted.values()}.isdisjoint({os.getpid()})
        assert {draws[3] for draws in fitted.values()} == {1}

    def test_one_job_stays_in_the_calling_process(self) -> None:
        fitted = run_per_drug(_draw, _tasks("d1", "d2"), n_jobs=1)

        assert {draws[4] for draws in fitted.values()} == {os.getpid()}

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_an_error_for_one_drug_is_raised_for_all(self, n_jobs: int) -> None:
        with pytest.raises(RuntimeError, match="d2 does not converge"):
            run_per_drug(_fail_on_d2, [("d1", 1), ("d2", 2), ("d3", 3)], n_jobs=n_jobs)
