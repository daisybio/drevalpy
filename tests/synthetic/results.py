"""Result-object factories for tests that need a populated experiment.

The production path builds :class:`~drevalpy.types.results.run.RunResult` objects
inside the training loop, so anything that consumes results - the visualization
plots, the report writer, the result serializers - would otherwise need a full
training run to get an input. These factories produce the same shapes directly.

The defaults are chosen so a bare :func:`make_experiment_result` satisfies every
plot requirement in the package: three models (``critical_difference`` feeds
``scipy.stats.friedmanchisquare``, which raises below three), equal fold counts
across models, and enough pairs per fold for ``regression_scatter`` to have at
least two rows per group. One model is named ``NaiveMeanEffectsPredictor`` so
:meth:`~drevalpy.types.results.experiment.ExperimentResult.normalize` finds its
default reference.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.run import RunResult

#: ``normalize()`` uses this name as its default reference model.
REFERENCE_MODEL: Final = "NaiveMeanEffectsPredictor"

#: Three models keeps ``critical_difference`` (Friedman test) constructible.
DEFAULT_MODEL_NAMES: Final = (REFERENCE_MODEL, "ElasticNet", "RandomForest")

DEFAULT_DATASET_NAME: Final = "SyntheticDataset"
DEFAULT_SPLIT_MODE: Final = "LPO"

#: Suffix older drevalpy releases appended to the normalized copy of a metric.
#: ``normalize()`` no longer emits it - it recomputes every metric under its
#: plain name - so the builders below do not either; the constant stays for the
#: tests that pin the plots' tolerance of results written by those releases.
NORMALIZED_METRIC: Final = "Pearson: normalized"


def make_metrics(*, seed: int = 0) -> dict[str, float]:
    """Build a metrics dict covering every metric the package reports.

    Args:
        seed: Seed for the deterministic pseudo-random values.

    Returns:
        Mapping of metric name to score, holding every key in
        :data:`drevalpy.evaluation.AVAILABLE_METRICS` - the same key set a run
        carries in production, before and after normalization.
    """
    rng = np.random.default_rng(seed)
    return {name: float(rng.uniform(0.1, 0.9)) for name in AVAILABLE_METRICS}


def make_run_result(
    *,
    model_name: str = "ElasticNet",
    dataset_name: str = DEFAULT_DATASET_NAME,
    fold_index: int = 0,
    fold_id: str | None = None,
    split_mode: str = DEFAULT_SPLIT_MODE,
    n_pairs: int = 20,
    n_cell_lines: int = 5,
    n_drugs: int = 4,
    metrics: dict[str, float] | None = None,
    best_hyperparameters: dict[str, Any] | None = None,
    fold_metadata: dict[str, Any] | None = None,
    randomization: tuple[str, str] | None = None,
    seed: int | None = None,
) -> RunResult:
    """Build one fold's worth of predictions for a single model.

    ``cell_line_ids`` and ``drug_ids`` are cycled independently over the
    requested cardinalities, so every pair is unique as long as
    ``n_pairs <= n_cell_lines * n_drugs`` and the two counts are coprime; the
    defaults satisfy both.

    Args:
        model_name: Value for ``RunResult.model_name``.
        dataset_name: Value for ``RunResult.dataset_name``. Every run in one
            experiment must agree on this.
        fold_index: Zero-based fold number.
        fold_id: Value for ``RunResult.fold_id``. Defaults to
            ``f"fold_{fold_index}"``, which is what ``normalize()`` matches runs
            on across models.
        split_mode: Value for ``RunResult.split_mode``. Every run in one
            experiment must agree on this.
        n_pairs: Number of cell-line/drug pairs in the fold.
        n_cell_lines: Number of distinct cell-line ids to cycle through.
        n_drugs: Number of distinct drug ids to cycle through.
        metrics: Metrics dict. Defaults to :func:`make_metrics`.
        best_hyperparameters: Value for ``RunResult.best_hyperparameters``.
        fold_metadata: Value for ``RunResult.fold_metadata``. Add a
            ``"robustness_trial"`` key here to make an experiment report
            ``has_robustness``.
        randomization: Value for ``RunResult.randomization``. Set it to make an
            experiment report ``has_randomization``.
        seed: Seed for predictions, ground truth and default metrics. Defaults
            to ``fold_index``, so folds of one model differ but the same fold of
            two models does not.

    Returns:
        A fully populated ``RunResult``.
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
        cell_line_ids=np.array([f"CL_{i % n_cell_lines}" for i in range(n_pairs)]),
        drug_ids=np.array([f"D_{i % n_drugs}" for i in range(n_pairs)]),
        split_mode=split_mode,
        fold_id=f"fold_{fold_index}" if fold_id is None else fold_id,
        best_hyperparameters=dict(best_hyperparameters or {"alpha": 0.1}),
        metrics=make_metrics(seed=effective_seed) if metrics is None else dict(metrics),
        fold_metadata=dict(fold_metadata or {"fold_index": fold_index}),
        randomization=randomization,
    )


def make_model_result(
    *,
    model_name: str = "ElasticNet",
    dataset_name: str = DEFAULT_DATASET_NAME,
    n_folds: int = 3,
    split_mode: str = DEFAULT_SPLIT_MODE,
    n_pairs: int = 20,
) -> ModelResult:
    """Build one model's folds.

    Args:
        model_name: Value for ``ModelResult.model_name``.
        dataset_name: Value for ``ModelResult.dataset_name``.
        n_folds: Number of runs to generate.
        split_mode: ``split_mode`` for every generated run.
        n_pairs: Number of pairs per fold.

    Returns:
        A ``ModelResult`` holding ``n_folds`` runs.
    """
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
    """Build a complete experiment with equal fold counts across models.

    Args:
        n_models: Number of models. Ignored when ``model_names`` is given. Names
            beyond :data:`DEFAULT_MODEL_NAMES` are generated as ``"Model_{i}"``.
        n_folds: Number of folds per model. Every model gets the same count,
            which ``critical_difference`` requires.
        model_names: Explicit model names. The first entry should be
            :data:`REFERENCE_MODEL` if the caller intends to call
            ``normalize()``.
        dataset_name: Shared ``dataset_name`` for every run.
        split_mode: Shared ``split_mode`` for every run.
        n_pairs: Number of pairs per fold.
        with_randomization: Attach randomization metadata to every run, making
            the experiment report ``has_randomization``.
        with_robustness: Attach a ``"robustness_trial"`` key to every run's
            ``fold_metadata``, making the experiment report ``has_robustness``.

    Returns:
        An ``ExperimentResult`` grouping ``n_models`` models of ``n_folds``
        folds each.
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
            fold_metadata=({"fold_index": fold_index, "robustness_trial": fold_index} if with_robustness else None),
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
