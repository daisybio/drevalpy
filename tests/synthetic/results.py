"""Result-object factories standing in for a completed training run.

Defaults satisfy every plot requirement: three models (Friedman needs three), equal fold counts,
enough pairs per fold, and ``NaiveMeanEffectsPredictor`` as the default normalization reference.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from drevalpy.evaluation import Metric
from drevalpy.types.results._identity import RunIdentity
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.failure import RunFailure
from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.run import RunResult

#: ``normalize()`` uses this name as its default reference model.
REFERENCE_MODEL: Final = "NaiveMeanEffectsPredictor"

#: Three models keeps ``critical_difference`` (Friedman test) constructible.
DEFAULT_MODEL_NAMES: Final = (REFERENCE_MODEL, "ElasticNet", "RandomForest")

DEFAULT_DATASET_NAME: Final = "SyntheticDataset"
DEFAULT_SPLIT_MODE: Final = "LPO"


def make_metrics(*, seed: int = 0) -> dict[str, float]:
    """A metrics dict holding every metric name, deterministic per *seed*."""
    rng = np.random.default_rng(seed)
    return {name: float(rng.uniform(0.1, 0.9)) for name in Metric}


def make_run_result(
    *,
    model_name: str = "ElasticNet",
    dataset_name: str = DEFAULT_DATASET_NAME,
    fold_index: int = 0,
    fold_id: str | None = None,
    split_mode: str = DEFAULT_SPLIT_MODE,
    n_pairs: int = 20,
    n_samples: int = 5,
    n_drugs: int = 4,
    metrics: dict[str, float] | None = None,
    best_hyperparameters: dict[str, Any] | None = None,
    fold_metadata: dict[str, Any] | None = None,
    randomization: tuple[str, str] | None = None,
    seed: int | None = None,
) -> RunResult:
    """Build one fold of predictions for a model.

    Sample and drug ids cycle independently, so pairs are unique when ``n_pairs <= n_samples * n_drugs`` and the
    counts are coprime. ``fold_id`` defaults to ``fold_{fold_index}``, which ``normalize()`` matches on; ``seed``
    defaults to ``fold_index``. Add ``"robustness_repeat"`` to *fold_metadata* or set *randomization* to make an
    experiment report ``has_robustness`` / ``has_randomization``.
    """
    effective_seed = fold_index if seed is None else seed
    rng = np.random.default_rng(effective_seed)
    ground_truth = rng.normal(size=n_pairs)
    predictions = ground_truth + rng.normal(scale=0.3, size=n_pairs)

    return RunResult(
        model_name=model_name,
        dataset_name=dataset_name,
        fold_index=fold_index,
        predictions=predictions,
        ground_truth=ground_truth,
        sample_ids=np.array([f"CL_{i % n_samples}" for i in range(n_pairs)]),
        drug_ids=np.array([f"D_{i % n_drugs}" for i in range(n_pairs)]),
        split_mode=split_mode,
        fold_id=f"fold_{fold_index}" if fold_id is None else fold_id,
        best_hyperparameters=dict(best_hyperparameters or {"alpha": 0.1}),
        metrics=make_metrics(seed=effective_seed) if metrics is None else dict(metrics),
        fold_metadata=dict(fold_metadata or {"fold_index": fold_index}),
        randomization=randomization,
    )


def make_run_failure(
    model_name: str = "ElasticNet",
    fold_index: int = 0,
    randomization: tuple[str, str] | None = None,
    robustness_repeat: int | None = None,
    error_type: str = "RuntimeError",
    message: str = "boom",
) -> RunFailure:
    """Build the record of one failed run."""
    identity = RunIdentity(
        model_name,
        DEFAULT_DATASET_NAME,
        DEFAULT_SPLIT_MODE,
        fold_index,
        randomization=randomization,
        robustness_repeat=robustness_repeat,
    )
    return RunFailure(identity, error_type, message)


def make_model_result(
    *,
    model_name: str = "ElasticNet",
    dataset_name: str = DEFAULT_DATASET_NAME,
    n_folds: int = 3,
    split_mode: str = DEFAULT_SPLIT_MODE,
    n_pairs: int = 20,
) -> ModelResult:
    """Build one model's ``n_folds`` runs."""
    return ModelResult(
        model_name=model_name,
        dataset_name=dataset_name,
        runs=[
            make_run_result(
                model_name=model_name,
                dataset_name=dataset_name,
                fold_index=fold_index,
                split_mode=split_mode,
                n_pairs=n_pairs,
            )
            for fold_index in range(n_folds)
        ],
    )


def make_experiment_result(
    *,
    n_models: int = 3,
    n_folds: int = 3,
    model_names: tuple[str, ...] | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split_mode: str = DEFAULT_SPLIT_MODE,
    n_pairs: int = 20,
    with_randomization: bool = False,
    with_robustness: bool = False,
) -> ExperimentResult:
    """Build an experiment with equal fold counts across models.

    Names beyond :data:`DEFAULT_MODEL_NAMES` are ``Model_{i}``; put :data:`REFERENCE_MODEL` first in *model_names* if
    the caller will ``normalize()``.
    """
    names = _resolve_model_names(n_models) if model_names is None else model_names

    runs = [
        make_run_result(
            model_name=name,
            dataset_name=dataset_name,
            fold_index=fold_index,
            split_mode=split_mode,
            n_pairs=n_pairs,
            seed=model_index * 100 + fold_index,
            fold_metadata=({"fold_index": fold_index, "robustness_repeat": fold_index} if with_robustness else None),
            randomization=("gene_expression", "permutation") if with_randomization else None,
        )
        for model_index, name in enumerate(names)
        for fold_index in range(n_folds)
    ]
    return ExperimentResult(runs)


def _resolve_model_names(n_models: int) -> tuple[str, ...]:
    if n_models <= len(DEFAULT_MODEL_NAMES):
        return DEFAULT_MODEL_NAMES[:n_models]
    extra = tuple(f"Model_{i}" for i in range(len(DEFAULT_MODEL_NAMES), n_models))
    return DEFAULT_MODEL_NAMES + extra
