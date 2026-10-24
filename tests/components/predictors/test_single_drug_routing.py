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
