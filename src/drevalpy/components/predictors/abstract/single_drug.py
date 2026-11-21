"""Base class for predictors that fit one model per drug.

A single-drug predictor reads no drug features and keeps one fitted model per drug. Subclasses
implement two hooks on a batch holding one drug's pairs: ``_fit_drug`` returns the fitted model
(or ``None`` to skip the drug) and ``_predict_drug`` scores that drug's pairs. The base routes
the pairs by drug, gives each drug its own checkpoint directory, fits the drugs in
``batch.training_context.n_jobs`` worker processes, seeds every drug from the run seed and its
ID, and predicts NaN for the pairs of a drug it has no model for.

A drug's batch keeps only the sample and drug rows its pairs and early-stopping pairs use, so
what is sent to a worker grows with the drug's pairs rather than with the whole entity table.
"""

from __future__ import annotations

import hashlib
import random
from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from upath import UPath

from drevalpy._utils.seed import drug_seed, seed_everything
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope


class SingleDrugPredictor(Predictor):
    """Fit and score one model per drug through ``_fit_drug`` and ``_predict_drug``."""

    drug_inputs: ClassVar[Inputs] = Inputs.none()
    scope: ClassVar[ModelScope] = ModelScope.SINGLE_DRUG

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the predictor without per-drug models.

        :param hyperparameters: Optional hyperparameter overrides, passed to the base class.
        """
        super().__init__(hyperparameters)
        self._models: dict[str, Any] = {}

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit one model per drug of *batch*, in ``batch.training_context.n_jobs`` worker processes.

        Errors from ``_fit_drug`` propagate; one failing drug fails the whole fit.

        :param batch: Training batch.
        """
        tasks = [(drug_id, _drug_batch(batch, drug_id, mask)) for drug_id, mask in _drug_masks(batch)]
        fitted = _run_per_drug(self._fit_drug, tasks, batch.training_context.n_jobs)
        self._models = {drug_id: model for drug_id, model in fitted.items() if model is not None}

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Score every pair with the model of its drug.

        :param batch: Pairs to score.
        :returns: One prediction per pair; NaN for pairs of drugs without a model.
        """
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id, mask in _drug_masks(batch):
            model = self._models.get(drug_id)
            if model is not None:
                drug_predictions = self._predict_drug(drug_id, model, _drug_batch(batch, drug_id, mask))
                predictions[mask] = np.asarray(drug_predictions, dtype=np.float64).ravel()
        return predictions

    @abstractmethod
    def _fit_drug(self, drug_id: str, batch: ModelInputBatch) -> Any | None:
        """Fit the model of one drug.

        Runs in a worker process when ``n_jobs`` is above 1, so it must return the model rather
        than store it on ``self``, and must not start processes of its own.

        :param drug_id: Drug whose pairs *batch* holds.
        :param batch: The drug's training pairs, with a checkpoint directory of its own in
            ``batch.training_context``.
        :returns: The fitted model, or ``None`` to leave the drug without a model.
        """

    @abstractmethod
    def _predict_drug(self, drug_id: str, model: Any, batch: ModelInputBatch) -> np.ndarray:
        """Score the pairs of one drug.

        :param drug_id: Drug whose pairs *batch* holds.
        :param model: What ``_fit_drug`` returned for the drug.
        :param batch: The drug's pairs to score.
        :returns: One prediction per pair of *batch*.
        """


def _checkpoint_dir_for_drug(base_dir: UPath | str, drug_id: str) -> UPath:
    """Return a unique checkpoint directory for a given drug.

    Hashing keeps the segment filesystem-safe: drug ids come from the dataset and
    routinely contain ``/`` and other separators.

    :param base_dir: Base directory for checkpoints.
    :param drug_id: Drug identifier to hash.
    :returns: Path to the drug-specific checkpoint directory.
    """
    digest = hashlib.sha256(drug_id.encode()).hexdigest()[:16]
    return UPath(base_dir) / f"drug_{digest}"


def _drug_masks(batch: ModelInputBatch) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(drug_id, pair_mask)`` for each drug of *batch*, in sorted drug order.

    :param batch: Batch to route.
    :yields: Drug identifier and boolean mask over the pairs.
    """
    keys = np.asarray(batch.drug_ids, dtype=str)
    for drug_id in np.unique(keys):
        yield str(drug_id), keys == drug_id


def _drug_batch(batch: ModelInputBatch, drug_id: str, mask: np.ndarray) -> ModelInputBatch:
    """Cut the pairs of one drug out of *batch*, with the drug's own single-process context.

    :param batch: Batch holding every drug.
    :param drug_id: Drug the pairs of *mask* belong to.
    :param mask: The drug's pairs.
    :returns: The drug's pairs over entity tables reduced to the rows they use.
    """
    context = TrainingContext(
        checkpoint_dir=_checkpoint_dir_for_drug(batch.training_context.checkpoint_dir, drug_id), n_jobs=1
    )
    sub_batch = batch.subset_pairs(mask)
    stopping = sub_batch.early_stopping_response
    sample, sample_pair_idx = _keep_used_rows(
        sub_batch.sample, sub_batch.sample_pair_idx, None if stopping is None else stopping.sample_ids
    )
    drug, drug_pair_idx = sub_batch.drug, sub_batch.drug_pair_idx
    if drug is not None and drug_pair_idx is not None:
        drug, drug_pair_idx = _keep_used_rows(drug, drug_pair_idx, None if stopping is None else stopping.drug_ids)
    return replace(
        sub_batch,
        sample=sample,
        drug=drug,
        sample_pair_idx=sample_pair_idx,
        drug_pair_idx=drug_pair_idx,
        training_context=context,
    )


def _keep_used_rows(
    table: FeaturizerResult, pair_idx: np.ndarray, extra_ids: np.ndarray | None
) -> tuple[FeaturizerResult, np.ndarray]:
    """Drop the entity rows that neither a pair nor an early-stopping pair uses.

    :param table: Entity table the pairs index into.
    :param pair_idx: Row of each pair in *table*.
    :param extra_ids: Further entities to keep, such as the early-stopping ones.
    :returns: The reduced table and each pair's row in it.
    """
    keep = np.zeros(len(table.entity_ids), dtype=bool)
    keep[pair_idx] = True
    if extra_ids is not None:
        keep |= np.isin(table.entity_ids, extra_ids)
    blocks = {
        name: replace(block, values=block.values[keep]) if block.entity_aligned else block
        for name, block in table.blocks.items()
    }
    new_rows = np.cumsum(keep) - 1
    return FeaturizerResult(table.entity_ids[keep], blocks, table.projection), new_rows[pair_idx]


def _run_per_drug(
    fit_one: Callable[[str, ModelInputBatch], Any],
    tasks: list[tuple[str, ModelInputBatch]],
    n_jobs: int,
) -> dict[str, Any]:
    """Fit one model per drug, optionally in parallel worker processes.

    With several drugs, every one is seeded from its id and a root seed drawn from the global RNG,
    in the calling process as well as in a worker, and the global RNGs are left in the same state
    afterwards. The fitted models and everything that follows therefore do not depend on
    ``n_jobs``. A lone drug has nothing to race with: it is fitted in the calling process with
    the RNG state it finds. With ``n_jobs > 1`` a worker runs single-threaded, so a forest with
    ``n_jobs=-1`` or a BLAS call does not start one thread per CPU in each of them. An error for
    one drug is raised for the whole call. ``fit_one``, the tasks and the results must be picklable.

    :param fit_one: Called as ``fit_one(drug_id, batch)``; returns the fitted model of the drug.
    :param tasks: ``(drug_id, batch)`` per drug.
    :param n_jobs: Number of worker processes; ``1`` fits in the calling process and ``-1`` uses all CPUs.
    :returns: The fitted model per drug, in the order of ``tasks``.
    """
    if len(tasks) <= 1:
        return {drug_id: fit_one(drug_id, task) for drug_id, task in tasks}

    root_seed = random.getrandbits(32)
    if n_jobs == 1:
        results = [_fit_seeded(fit_one, drug_seed(root_seed, drug_id), drug_id, task) for drug_id, task in tasks]
    else:
        import joblib

        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_fit_in_worker)(fit_one, drug_seed(root_seed, drug_id), drug_id, task)
                for drug_id, task in tasks
            )
    seed_everything(root_seed)
    return dict(zip((drug_id for drug_id, _ in tasks), results, strict=True))


def _fit_seeded(fit_one: Callable[[str, ModelInputBatch], Any], seed: int, drug_id: str, task: ModelInputBatch) -> Any:
    """Seed every RNG, then fit one drug.

    :param fit_one: Fit function of :func:`_run_per_drug`.
    :param seed: Seed of this drug.
    :param drug_id: Drug to fit.
    :param task: Its batch.
    :returns: The fitted model.
    """
    seed_everything(seed)
    return fit_one(drug_id, task)


def _fit_in_worker(
    fit_one: Callable[[str, ModelInputBatch], Any], seed: int, drug_id: str, task: ModelInputBatch
) -> Any:
    """Fit one drug in a worker process without nested parallelism.

    :param fit_one: Fit function of :func:`_run_per_drug`.
    :param seed: Seed of this drug.
    :param drug_id: Drug to fit.
    :param task: Its batch.
    :returns: The fitted model.
    """
    import joblib
    import torch

    torch.set_num_threads(1)
    with joblib.parallel_config(backend="sequential"):
        return _fit_seeded(fit_one, seed, drug_id, task)
