"""Shared drug-id routing and per-drug fitting for single-drug predictors."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

import numpy as np

from drevalpy._utils.seed import drug_seed, seed_everything
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

_Task = TypeVar("_Task")
_Fitted = TypeVar("_Fitted")


def routing_keys(batch: ModelInputBatch) -> np.ndarray:
    """Return the drug id of every pair, the key a single-drug predictor routes it by.

    :param batch: Batch to route.
    :returns: Drug id string per pair.
    """
    return np.asarray(batch.drug_ids, dtype=str)


def iter_drug_masks(batch: ModelInputBatch) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(drug_id, pair_mask)`` for each known drug in the batch.

    :param batch: Batch to route.
    :yields: Drug identifier and boolean mask over response pairs.
    """
    keys = routing_keys(batch)
    for drug_id in np.unique(keys):
        yield str(drug_id), keys == drug_id


def run_per_drug(
    fit_one: Callable[[str, _Task], _Fitted],
    tasks: Iterable[tuple[str, _Task]],
    n_jobs: int = 1,
) -> dict[str, _Fitted]:
    """Fit one model per drug, optionally in parallel worker processes.

    With several drugs, every one is seeded from its id and a root seed drawn from the global RNG,
    in the calling process as well as in a worker, and the global RNGs are left in the same state
    afterwards. The fitted models and everything that follows therefore do not depend on
    ``n_jobs``. A lone drug has nothing to race with: it is fitted in the calling process with
    the RNG state it finds. With ``n_jobs > 1`` a worker runs single-threaded, so a forest with
    ``n_jobs=-1`` or a BLAS call does not start one thread per CPU in each of them. An error for
    one drug is raised for the whole call. ``fit_one``, the tasks and the results must be picklable.

    :param fit_one: Called as ``fit_one(drug_id, task)``; returns the fitted model of the drug.
    :param tasks: ``(drug_id, task)`` per drug.
    :param n_jobs: Number of worker processes; ``1`` fits in the calling process and ``-1`` uses all CPUs.
    :returns: The fitted model per drug, in the order of ``tasks``.
    """
    jobs = list(tasks)
    if len(jobs) <= 1:
        return {drug_id: fit_one(drug_id, task) for drug_id, task in jobs}

    root_seed = random.getrandbits(32)
    if n_jobs == 1:
        results = [_fit_seeded(fit_one, drug_seed(root_seed, drug_id), drug_id, task) for drug_id, task in jobs]
    else:
        import joblib

        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_fit_in_worker)(fit_one, drug_seed(root_seed, drug_id), drug_id, task)
                for drug_id, task in jobs
            )
    seed_everything(root_seed)
    return dict(zip((drug_id for drug_id, _ in jobs), results, strict=True))


def _fit_seeded(fit_one: Callable[[str, _Task], _Fitted], seed: int, drug_id: str, task: _Task) -> _Fitted:
    """Seed every RNG, then fit one drug.

    :param fit_one: Fit function of :func:`run_per_drug`.
    :param seed: Seed of this drug.
    :param drug_id: Drug to fit.
    :param task: Its task.
    :returns: The fitted model.
    """
    seed_everything(seed)
    return fit_one(drug_id, task)


def _fit_in_worker(fit_one: Callable[[str, _Task], _Fitted], seed: int, drug_id: str, task: _Task) -> _Fitted:
    """Fit one drug in a worker process without nested parallelism.

    :param fit_one: Fit function of :func:`run_per_drug`.
    :param seed: Seed of this drug.
    :param drug_id: Drug to fit.
    :param task: Its task.
    :returns: The fitted model.
    """
    import joblib
    import torch

    torch.set_num_threads(1)
    with joblib.parallel_config(backend="sequential"):
        return _fit_seeded(fit_one, seed, drug_id, task)
