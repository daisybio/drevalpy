"""Top-level pipeline orchestrating models x folds x randomization.

Stress tests do not tune: every randomized dataset and every robustness fold is fitted with the
best hyperparameters of the original dataset on the original fold of the same index, so a stress
run differs from its original run only in the data. A robustness fold follows its original fold
in the fold list, and the original dataset comes first among the datasets, so that run has always
finished before its variants start; when it failed, the variants are recorded as failures.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from rich.markup import escape

from drevalpy._single import _fit_fold
from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.data import CurveQuality, split
from drevalpy.data._resolve import resolve_dataset
from drevalpy.errors import DrEvalError, InvalidInputError
from drevalpy.experiment._execution import check_mode_supported
from drevalpy.experiment._randomization import randomization
from drevalpy.experiment._robustness import robustness
from drevalpy.log import get_logger
from drevalpy.models.spec import ModelSpec
from drevalpy.registry.splitter import canonical
from drevalpy.types import Dataset, Fold
from drevalpy.types.enums.randomization import RandomizationType
from drevalpy.types.results import ExperimentResult, RunResult
from drevalpy.types.results._identity import RunIdentity
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer

logger = get_logger(__name__)

_T = TypeVar("_T")


class OriginalRunFailedError(DrEvalError, RuntimeError):
    """The original run of a fold failed, so its stress-test runs have no hyperparameters to reuse."""


def _validate_models(models: list[ModelSpec]) -> None:
    if any(not isinstance(model, ModelSpec) for model in models):
        raise TypeError("run() expects ModelSpec values")
    if len({model.name for model in models}) != len(models):
        raise InvalidInputError("Model names must be unique within an experiment")


@dataclass
class _Outcomes:
    """The runs that finished and the failures that were recorded, under one error policy."""

    on_error: str
    dataset_name: str
    runs: list[RunResult] = field(default_factory=list)
    failures: list[RunFailure] = field(default_factory=list)

    def attempt(
        self,
        call: Callable[[], _T],
        *,
        model_name: str,
        folds: Sequence[Fold],
        randomized: tuple[str, str] | None = None,
    ) -> _T | None:
        """Return the result of *call*; on failure record one failure per fold unless ``on_error`` is ``"raise"``."""
        try:
            return call()
        except Exception as error:
            if self.on_error == "raise":
                raise
            for fold in folds:
                self.record(error, model_name=model_name, fold=fold, randomized=randomized)
            return None

    def fit_folds(
        self,
        fit: Callable[..., RunResult],
        model: ModelSpec,
        datasets: Sequence[Dataset],
        folds: Sequence[Fold],
    ) -> None:
        """Fit *model* on every fold of every dataset (original first, then its randomizations).

        The original dataset on an original fold is tuned; every other run reuses its best hyperparameters.

        :param fit: ``_fit_fold`` bound to the model and run settings; called with a dataset, a fold and
            ``hyperparameters``.
        :param model: Model being fitted, for the failure records.
        :param datasets: The original dataset, then its randomized copies.
        :param folds: Original folds, each followed by its robustness copies.
        """
        best_by_fold: dict[int, Mapping[str, Any]] = {}
        for fold in folds:
            original_fold = "robustness_repeat" not in fold.metadata
            for position, run_ds in enumerate(datasets):
                self._fit_run(fit, model.name, run_ds, fold, best_by_fold, original=original_fold and position == 0)

    def _fit_run(
        self,
        fit: Callable[..., RunResult],
        model_name: str,
        run_ds: Dataset,
        fold: Fold,
        best_by_fold: dict[int, Mapping[str, Any]],
        *,
        original: bool,
    ) -> None:
        """Fit one run: tune an *original* run and keep its values, or reuse them for a stress-test run."""
        fold_index = fold.metadata.get("fold_index", 0)
        if not original and fold_index not in best_by_fold:
            error = OriginalRunFailedError(
                f"the original run of fold {fold_index} failed, so this stress-test run was skipped"
            )
            self.record(error, model_name=model_name, fold=fold, randomized=run_ds.randomization)
            return
        hyperparameters = None if original else best_by_fold[fold_index]
        result = self.attempt(
            partial(fit, run_ds, fold, hyperparameters=hyperparameters),
            model_name=model_name,
            folds=[fold],
            randomized=run_ds.randomization,
        )
        if result is None:
            return
        self.runs.append(result)
        if original:
            best_by_fold[fold_index] = result.best_hyperparameters

    def record(
        self,
        error: Exception,
        *,
        model_name: str,
        fold: Fold,
        randomized: tuple[str, str] | None = None,
    ) -> None:
        """Store and log a failure of the run on *fold*."""
        metadata = fold.metadata
        identity = RunIdentity.from_metadata(
            model_name,
            self.dataset_name,
            canonical(metadata.get("split_mode", "")),
            metadata.get("fold_index", 0),
            metadata.get("fold_id", ""),
            randomized,
            metadata,
        )
        failure = RunFailure.from_exception(error, identity)
        logger.warning("Run failed, continuing: %s", escape(str(failure)))
        self.failures.append(failure)


def run(
    models: list[ModelSpec],
    dataset: Dataset | str | Sequence[Dataset | str],
    split_mode: str,
    randomization_modes: list[str] | None = None,
    randomization_type: RandomizationType = RandomizationType.PERMUTATION,
    hyperparameter_tuning: bool = True,
    hpo_metric: str = "RMSE",
    hpo_num_trials: int = 16,
    hpo_random_state: int | None = None,
    robustness_repeats: int = 0,
    randomization_random_state: int | None = None,
    seed: int = 42,
    response_transformation: ResponseTransformer | None = None,
    quality: CurveQuality | None = None,
    measure: str | None = None,
    wandb_project: str | None = None,
    on_error: Literal["record", "raise"] = "record",
    n_jobs: int = 1,
    n_splits: int = 5,
) -> ExperimentResult:
    """Run the full experiment pipeline.

    :param models: Named model specifications to evaluate.
    :param dataset: Dataset object or name to load, or a sequence of them to concatenate into one
        cross-study dataset (see ``drevalpy.data.concat``). A sequence of one is used as it is.
    :param split_mode: Split mode (LPO, LSO, LDO, LTO, LStO).
    :param randomization_modes: Optional randomization modes (SVRC, SVCC, SVRD, SVCD).
    :param randomization_type: How to replace the view.
    :param hyperparameter_tuning: Whether to run HPO. Only the original dataset on each original fold is tuned; its
        randomized datasets and robustness folds reuse the best hyperparameters found there.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_trials: Number of HPO trials.
    :param hpo_random_state: Random seed for the HPO sampler; defaults to ``seed``.
    :param robustness_repeats: Robustness repeats per fold (0 = none).
    :param randomization_random_state: Root seed for reproducible feature randomization; defaults to ``seed``.
    :param seed: Root seed driving the splits, the HPO and randomization defaults, and every model fit.
    :param response_transformation: Optional unfitted response transformation, cloned for every fit. The normalization
        reference ``NaiveMeanEffectsPredictor`` is always fitted on the untransformed response.
    :param quality: Curve-quality rule passed to the splitter; ``None`` means ``CurveQuality.default()``.
    :param measure: Response measure to train and score on, such as ``LN_IC50`` or ``AUC``: ``pEC50`` or any
        response layer of the dataset (see :meth:`~drevalpy.types.data.dataset.Dataset.with_measure`). ``None``
        keeps the dataset's measure, pEC50 unless it was set.
    :param wandb_project: Weights & Biases project to log to; ``None`` disables tracking.
    :param on_error: ``"record"`` keeps going when a run raises, logs a warning and stores a ``RunFailure`` on
        the result; ``"raise"`` re-raises the first error. ``KeyboardInterrupt`` always propagates.
    :param n_jobs: Number of worker processes a single-drug model fits its per-drug models in, during tuning and in
        every final fit; ``-1`` uses all CPUs. Results are identical for every value, and other models ignore it.
    :param n_splits: Number of CV folds the splitter makes; the k-fold splitters reject fewer than 2. LStO
        ignores it and makes one fold per study.
    :returns: ExperimentResult grouping all model results, with the recorded failures.
    :raises ExperimentFailedError: If ``on_error="record"`` and no run produced a result.
    :raises InvalidInputError: If ``on_error`` is neither ``"record"`` nor ``"raise"``, or ``n_jobs`` is neither
        positive nor ``-1``. A model that *split_mode* cannot score, such as a single-drug model under LDO, raises
        ``InvalidInputError`` from ``check_mode_supported`` before the dataset is loaded.
    """
    check_n_jobs(n_jobs)
    if on_error not in ("record", "raise"):
        raise InvalidInputError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    _validate_models(models)
    for model in models:
        check_mode_supported(model, split_mode)
    ds = resolve_dataset(dataset)
    if measure is not None:
        ds = ds.with_measure(measure)
    folds = split(ds, split_mode, n_splits=n_splits, random_state=seed, quality=quality)

    if robustness_repeats > 0:
        folds = [s for fold in folds for s in [fold, *robustness(fold, robustness_repeats)]]

    randomization_seed = seed if randomization_random_state is None else randomization_random_state
    outcomes = _Outcomes(on_error, ds.name)

    fit = partial(
        _fit_fold,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        seed=seed,
        wandb_project=wandb_project,
        n_jobs=n_jobs,
    )

    for model in models:
        build = partial(
            randomization,
            model,
            ds,
            randomization_modes or [],
            randomization_type=randomization_type,
            random_state=randomization_seed,
        )
        randomized = outcomes.attempt(build, model_name=model.name, folds=folds) or []
        outcomes.fit_folds(
            partial(fit, model, response_transformation=response_transformation), model, [ds, *randomized], folds
        )

    if outcomes.failures and not outcomes.runs:
        raise ExperimentFailedError(outcomes.failures)
    return ExperimentResult(outcomes.runs, outcomes.failures)
