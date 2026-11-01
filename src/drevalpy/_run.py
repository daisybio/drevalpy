"""Top-level pipeline orchestrating models x folds x randomization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Literal, TypeVar

from rich.markup import escape

from drevalpy._single import _fit_fold
from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.data import CurveQuality, split
from drevalpy.data._resolve import resolve_dataset
from drevalpy.errors import InvalidInputError
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

#: The model every normalized metric is measured against (``ExperimentResult.normalize``). It always trains
#: on the original response scale, so a response transformation cannot move the ruler along with the models.
NORMALIZATION_REFERENCE = "NaiveMeanEffectsPredictor"

_T = TypeVar("_T")


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
        fit: Callable[[Dataset, Fold], RunResult],
        model: ModelSpec,
        datasets: Sequence[Dataset],
        folds: Sequence[Fold],
    ) -> None:
        """Fit *model* on every fold of every dataset (original first, then its randomizations)."""
        for fold in folds:
            for run_ds in datasets:
                result = self.attempt(
                    partial(fit, run_ds, fold),
                    model_name=model.name,
                    folds=[fold],
                    randomized=run_ds.randomization,
                )
                if result is not None:
                    self.runs.append(result)

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


def _transformation_for(
    model: ModelSpec, response_transformation: ResponseTransformer | None
) -> ResponseTransformer | None:
    """Return the transformation *model* trains with: none for the normalization reference."""
    if model.name == NORMALIZATION_REFERENCE and response_transformation is not None:
        logger.info("%s trains on the untransformed response: it is the normalization reference", model.name)
        return None
    return response_transformation


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
    wandb_project: str | None = None,
    on_error: Literal["record", "raise"] = "record",
    n_jobs: int = 1,
) -> ExperimentResult:
    """Run the full experiment pipeline.

    :param models: Named model specifications to evaluate.
    :param dataset: Dataset object or name to load, or a sequence of them to concatenate into one
        cross-study dataset (see ``drevalpy.data.concat``). A sequence of one is used as it is.
    :param split_mode: Split mode (LPO, LSO, LDO, LTO, LStO).
    :param randomization_modes: Optional randomization modes (SVRC, SVCC, SVRD, SVCD).
    :param randomization_type: How to replace the view.
    :param hyperparameter_tuning: Whether to run HPO.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_trials: Number of HPO trials.
    :param hpo_random_state: Random seed for the HPO sampler; defaults to ``seed``.
    :param robustness_repeats: Robustness repeats per fold (0 = none).
    :param randomization_random_state: Root seed for reproducible feature randomization; defaults to ``seed``.
    :param seed: Root seed driving the splits, the HPO and randomization defaults, and every model fit.
    :param response_transformation: Optional unfitted response transformation, cloned for every fit. The normalization
        reference ``NaiveMeanEffectsPredictor`` is always fitted on the untransformed response.
    :param quality: Curve-quality rule passed to the splitter; ``None`` means ``CurveQuality.default()``.
    :param wandb_project: Weights & Biases project to log to; ``None`` disables tracking.
    :param on_error: ``"record"`` keeps going when a run raises, logs a warning and stores a ``RunFailure`` on
        the result; ``"raise"`` re-raises the first error. ``KeyboardInterrupt`` always propagates.
    :param n_jobs: Number of worker processes a single-drug model fits its per-drug models in, during tuning and in
        every final fit; ``-1`` uses all CPUs. Results are identical for every value, and other models ignore it.
    :returns: ExperimentResult grouping all model results, with the recorded failures.
    :raises ExperimentFailedError: If ``on_error="record"`` and no run produced a result.
    :raises InvalidInputError: If ``on_error`` is neither ``"record"`` nor ``"raise"``, or ``n_jobs`` is neither
        positive nor ``-1``.
    """
    check_n_jobs(n_jobs)
    if on_error not in ("record", "raise"):
        raise InvalidInputError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    _validate_models(models)
    ds = resolve_dataset(dataset)
    folds = split(ds, split_mode, random_state=seed, quality=quality)

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
        transformation = _transformation_for(model, response_transformation)
        outcomes.fit_folds(partial(fit, model, response_transformation=transformation), model, [ds, *randomized], folds)

    if outcomes.failures and not outcomes.runs:
        raise ExperimentFailedError(outcomes.failures)
    return ExperimentResult(outcomes.runs, outcomes.failures)
