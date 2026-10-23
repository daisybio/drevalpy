"""Top-level pipeline orchestrating models x folds x randomization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

from rich.markup import escape

from drevalpy._single import single
from drevalpy.data import CurveQuality, concat, load, split
from drevalpy.errors import InvalidInputError
from drevalpy.experiment._randomization import randomization
from drevalpy.experiment._robustness import robustness
from drevalpy.log import get_logger
from drevalpy.models.spec import ModelSpec
from drevalpy.types import Dataset, SplitMasks
from drevalpy.types.results import ExperimentResult, RunResult
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure

logger = get_logger(__name__)


def _validate_models(models: list[ModelSpec]) -> None:
    if any(not isinstance(model, ModelSpec) for model in models):
        raise TypeError("run() expects ModelSpec values")
    if len({model.name for model in models}) != len(models):
        raise InvalidInputError("Model names must be unique within an experiment")


def _resolve_dataset(dataset: Dataset | str | Sequence[Dataset | str]) -> Dataset:
    """Load names, and concatenate a sequence of several datasets into one."""
    if isinstance(dataset, str) or not isinstance(dataset, Sequence):
        return load(dataset) if isinstance(dataset, str) else dataset
    if not dataset:
        raise InvalidInputError("dataset must not be an empty sequence")
    datasets = [load(item) if isinstance(item, str) else item for item in dataset]
    return datasets[0] if len(datasets) == 1 else concat(datasets)


@dataclass
class _Outcomes:
    """The runs that finished and the failures that were recorded, under one error policy."""

    on_error: str
    runs: list[RunResult] = field(default_factory=list)
    failures: list[RunFailure] = field(default_factory=list)

    def attempt(
        self,
        call: Callable[[], RunResult],
        *,
        model_name: str,
        masks: SplitMasks,
        randomized: tuple[str, str] | None,
    ) -> None:
        """Keep the result of *call*, or record its failure unless ``on_error`` is ``"raise"``."""
        try:
            self.runs.append(call())
        except Exception as error:
            if self.on_error == "raise":
                raise
            self.record(error, model_name=model_name, masks=masks, randomized=randomized)

    def build_randomized(
        self, build: Callable[[], list[Dataset]], *, model_name: str, folds: list[SplitMasks]
    ) -> list[Dataset]:
        """Build a model's randomized datasets; on failure record one failure per fold and build none."""
        try:
            return build()
        except Exception as error:
            if self.on_error == "raise":
                raise
            for masks in folds:
                self.record(error, model_name=model_name, masks=masks)
            return []

    def record(
        self,
        error: Exception,
        *,
        model_name: str,
        masks: SplitMasks,
        randomized: tuple[str, str] | None = None,
    ) -> None:
        """Store and log a failure of the run on *masks*."""
        failure = RunFailure.from_exception(
            error,
            model_name=model_name,
            fold_index=masks.metadata.get("fold_index", 0),
            randomization=randomized,
            robustness_trial=masks.metadata.get("robustness_trial"),
        )
        logger.warning("Run failed, continuing: %s", escape(str(failure)))
        self.failures.append(failure)


def _run_model(
    outcomes: _Outcomes,
    model: ModelSpec,
    ds: Dataset,
    folds: list[SplitMasks],
    fit: Callable[..., RunResult],
    randomize: Callable[..., list[Dataset]] | None,
    randomization_modes: list[str] | None,
) -> None:
    """Run *model* on every fold, on the dataset and on its randomized copies built once up front."""
    randomized_datasets = (
        outcomes.build_randomized(
            partial(randomize, model, ds, randomization_modes),
            model_name=model.name,
            folds=folds,
        )
        if randomize is not None
        else []
    )
    for masks in folds:
        for run_ds in [ds, *randomized_datasets]:
            outcomes.attempt(
                partial(fit, model, run_ds, masks),
                model_name=model.name,
                masks=masks,
                randomized=run_ds.randomization,
            )


def run(
    models: list[ModelSpec],
    dataset: Dataset | str | Sequence[Dataset | str],
    split_mode: str,
    randomization_modes: list[str] | None = None,
    randomization_type: str = "permutation",
    hyperparameter_tuning: bool = True,
    hpo_metric: str = "RMSE",
    hpo_num_trials: int = 16,
    hpo_random_state: int | None = None,
    robustness_trials: int = 0,
    randomization_random_state: int | None = None,
    seed: int = 42,
    response_transformation=None,
    quality: CurveQuality | None = None,
    wandb_project: str | None = None,
    on_error: Literal["record", "raise"] = "record",
) -> ExperimentResult:
    """Run the full experiment pipeline.

    :param models: Named model specifications to evaluate.
    :param dataset: Dataset object or name to load, or a sequence of them to concatenate into one
        cross-study dataset (see ``drevalpy.data.concat``). A sequence of one is used as it is.
    :param split_mode: Split mode (LPO, LSO, LDO, LTO, LStO).
    :param randomization_modes: Optional randomization modes (SVRC, SVCC, SVRD, SVCD).
    :param randomization_type: "permutation" or "invariant".
    :param hyperparameter_tuning: Whether to run HPO.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_trials: Number of HPO trials.
    :param hpo_random_state: Random seed for the HPO sampler; defaults to ``seed``.
    :param robustness_trials: Number of robustness permutations (0 = disabled).
    :param randomization_random_state: Root seed for reproducible feature randomization; defaults to ``seed``.
    :param seed: Root seed driving the splits, the HPO and randomization defaults, and every model fit.
    :param response_transformation: Optional unfitted target transformer, cloned for every fit.
    :param quality: Curve-quality rule applied before the splitter; ``None`` means ``CurveQuality.default()``.
    :param wandb_project: Weights & Biases project to log to; ``None`` disables tracking.
    :param on_error: ``"record"`` keeps going when a run raises, logs a warning and stores a ``RunFailure`` on
        the result; ``"raise"`` re-raises the first error. ``KeyboardInterrupt`` always propagates.
    :returns: ExperimentResult grouping all model results, with the recorded failures.
    :raises ExperimentFailedError: If ``on_error="record"`` and no run produced a result.
    """
    if on_error not in ("record", "raise"):
        raise InvalidInputError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    _validate_models(models)
    ds = _resolve_dataset(dataset)
    folds = split(ds, split_mode, random_state=seed, quality=quality)

    if robustness_trials > 0:
        folds = [s for fold in folds for s in [fold, *robustness(fold, robustness_trials)]]

    randomization_seed = seed if randomization_random_state is None else randomization_random_state
    outcomes = _Outcomes(on_error)

    fit = partial(
        single,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        seed=seed,
        response_transformation=response_transformation,
        wandb_project=wandb_project,
    )
    randomize = (
        partial(randomization, randomization_type=randomization_type, random_state=randomization_seed)
        if randomization_modes
        else None
    )
    for model in models:
        _run_model(outcomes, model, ds, folds, fit, randomize, randomization_modes)

    if outcomes.failures and not outcomes.runs:
        raise ExperimentFailedError(outcomes.failures)
    return ExperimentResult(outcomes.runs, outcomes.failures)
