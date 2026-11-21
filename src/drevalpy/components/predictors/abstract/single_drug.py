"""Base class for predictors that fit one model per drug.

A single-drug predictor reads no drug features and keeps one fitted model per drug. Subclasses
implement two hooks on an input holding one drug's pairs: ``_fit_drug`` returns the fitted model
(or ``None`` to skip the drug) and ``_predict_drug`` scores that drug's pairs. The base routes
the pairs by drug, gives each drug its own checkpoint directory, fits the drugs in
``data.training_context.n_jobs`` worker processes, seeds every drug from the run seed and its
ID, and predicts NaN for the pairs of a drug it has no model for.

A drug's input keeps only the sample and drug rows its pairs and early-stopping pairs use, so
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
from drevalpy.types.data.feature_block import FeaturizerOutput
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import ModelScope


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

    def _fit(self, data: PredictorInput) -> None:
        """Fit one model per drug of *data*, in ``data.training_context.n_jobs`` worker processes.

        Errors from ``_fit_drug`` propagate; one failing drug fails the whole fit.

        :param data: Training input.
        """
        tasks = [(drug_id, _drug_input(data, drug_id, mask)) for drug_id, mask in _drug_masks(data)]
        fitted = _run_per_drug(self._fit_drug, tasks, data.training_context.n_jobs)
        self._models = {drug_id: model for drug_id, model in fitted.items() if model is not None}

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Score every pair with the model of its drug.

        :param data: Pairs to score.
        :returns: One prediction per pair; NaN for pairs of drugs without a model.
        """
        predictions = np.full(len(data), np.nan, dtype=np.float64)
        for drug_id, mask in _drug_masks(data):
            model = self._models.get(drug_id)
            if model is not None:
                drug_predictions = self._predict_drug(drug_id, model, _drug_input(data, drug_id, mask))
                predictions[mask] = np.asarray(drug_predictions, dtype=np.float64).ravel()
        return predictions

    @abstractmethod
    def _fit_drug(self, drug_id: str, data: PredictorInput) -> Any | None:
        """Fit the model of one drug.

        Runs in a worker process when ``n_jobs`` is above 1, so it must return the model rather
        than store it on ``self``, and must not start processes of its own.

        :param drug_id: Drug whose pairs *data* holds.
        :param data: The drug's training pairs, with a checkpoint directory of its own in
            ``data.training_context``.
        :returns: The fitted model, or ``None`` to leave the drug without a model.
        """

    @abstractmethod
    def _predict_drug(self, drug_id: str, model: Any, data: PredictorInput) -> np.ndarray:
        """Score the pairs of one drug.

        :param drug_id: Drug whose pairs *data* holds.
        :param model: What ``_fit_drug`` returned for the drug.
        :param data: The drug's pairs to score.
        :returns: One prediction per pair of *data*.
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


def _drug_masks(data: PredictorInput) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(drug_id, pair_mask)`` for each drug of *data*, in sorted drug order.

    :param data: Model input to route.
    :yields: Drug identifier and boolean mask over the pairs.
    """
    keys = np.asarray(data.drug_ids, dtype=str)
    for drug_id in np.unique(keys):
        yield str(drug_id), keys == drug_id


def _drug_input(data: PredictorInput, drug_id: str, mask: np.ndarray) -> PredictorInput:
    """Cut the pairs of one drug out of *data*, with the drug's own single-process context.

    :param data: Input holding every drug.
    :param drug_id: Drug the pairs of *mask* belong to.
    :param mask: The drug's pairs.
    :returns: The drug's pairs over entity tables reduced to the rows they use.
    """
    context = TrainingContext(
        checkpoint_dir=_checkpoint_dir_for_drug(data.training_context.checkpoint_dir, drug_id), n_jobs=1
    )
    sub_data = data[mask]
    stopping = sub_data.early_stopping
    sample = _keep_used_rows(
        sub_data.sample, sub_data.sample_pair_idx, None if stopping is None else stopping.sample_ids
    )
    drug = sub_data.drug
    if drug is not None and sub_data.drug_pair_idx is not None:
        drug = _keep_used_rows(drug, sub_data.drug_pair_idx, None if stopping is None else stopping.drug_ids)
    return replace(
        sub_data,
        sample=sample,
        drug=drug,
        training_context=context,
    )


def _keep_used_rows(table: FeaturizerOutput, pair_idx: np.ndarray, extra_ids: np.ndarray | None) -> FeaturizerOutput:
    """Drop the entity rows that neither a pair nor an early-stopping pair uses.

    :param table: Entity table the pairs index into.
    :param pair_idx: Row of each pair in *table*.
    :param extra_ids: Further entities to keep, such as the early-stopping ones.
    :returns: The reduced entity table.
    """
    keep = np.zeros(len(table.entity_ids), dtype=bool)
    keep[pair_idx] = True
    if extra_ids is not None:
        keep |= np.isin(table.entity_ids, extra_ids)
    blocks = {
        name: replace(block, values=block.values[keep]) if block.entity_aligned else block
        for name, block in table.blocks.items()
    }
    return FeaturizerOutput(table.entity_ids[keep], blocks)


def _run_per_drug(
    fit_one: Callable[[str, PredictorInput], Any],
    tasks: list[tuple[str, PredictorInput]],
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

    :param fit_one: Called as ``fit_one(drug_id, data)``; returns the fitted model of the drug.
    :param tasks: ``(drug_id, data)`` per drug.
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


def _fit_seeded(fit_one: Callable[[str, PredictorInput], Any], seed: int, drug_id: str, task: PredictorInput) -> Any:
    """Seed every RNG, then fit one drug.

    :param fit_one: Fit function of :func:`_run_per_drug`.
    :param seed: Seed of this drug.
    :param drug_id: Drug to fit.
    :param task: Its input.
    :returns: The fitted model.
    """
    seed_everything(seed)
    return fit_one(drug_id, task)


def _fit_in_worker(fit_one: Callable[[str, PredictorInput], Any], seed: int, drug_id: str, task: PredictorInput) -> Any:
    """Fit one drug in a worker process without nested parallelism.

    :param fit_one: Fit function of :func:`_run_per_drug`.
    :param seed: Seed of this drug.
    :param drug_id: Drug to fit.
    :param task: Its input.
    :returns: The fitted model.
    """
    import joblib
    import torch

    torch.set_num_threads(1)
    with joblib.parallel_config(backend="sequential"):
        return _fit_seeded(fit_one, seed, drug_id, task)
