"""Top-level pipeline orchestrating models x folds x randomization.

Stress tests do not tune: every randomized dataset and every robustness fold is fitted with the
best hyperparameters of the original dataset on the original fold of the same index, so a stress
run differs from its original run only in the data. A robustness fold follows its original fold
in the fold list, and the original dataset comes first among the datasets, so that run has always
finished before its variants start; when it failed, the variants are recorded as failures.

With an ``output_dir``, every run is saved through a :class:`~drevalpy.types.results._store.RunStore` as soon as
it finishes, and a run that is already saved is loaded instead of fitted, so a killed run resumes where it
stopped. A loaded original run hands its best hyperparameters to its variants as a fitted one does. The store
compares the :func:`_settings` of the run with those the directory was started with; the dataset is identified
by its name and measure only, not by its content.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from rich.markup import escape

from drevalpy._run import _fit_fold
from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.data import CurveQuality, split
from drevalpy.data._resolve import resolve_dataset
from drevalpy.errors import InvalidInputError, OriginalRunFailedError
from drevalpy.execution._execution import check_mode_supported
from drevalpy.execution._randomization import randomization
from drevalpy.execution._robustness import robustness
from drevalpy.log import get_logger
from drevalpy.models.config.immutable import thaw_value
from drevalpy.models.spec import ModelSpec
from drevalpy.registry.splitter import canonical
from drevalpy.types import Dataset, Fold
from drevalpy.types.enums import RandomizationType
from drevalpy.types.results import ExperimentResult, RunResult
from drevalpy.types.results._identity import RunIdentity
from drevalpy.types.results._store import RunStore
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer

logger = get_logger(__name__)

_T = TypeVar("_T")


def _validate_models(models: list[ModelSpec]) -> None:
    if any(not isinstance(model, ModelSpec) for model in models):
        raise TypeError("experiment() expects ModelSpec values")
    if len({model.name for model in models}) != len(models):
        raise InvalidInputError("Model names must be unique within an experiment")


@dataclass
class _Outcomes:
    """The runs that finished and the failures that were recorded, under one error policy.

    With a *store*, every run is looked up in it before it is fitted and saved to it when it finishes.
    """

    on_error: str
    dataset_name: str
    store: RunStore | None = None
    runs: list[RunResult] = field(default_factory=list)
    failures: list[RunFailure] = field(default_factory=list)
    #: Best hyperparameters of each original run, by model name and fold index, for its stress-test runs.
    best_hyperparameters: dict[tuple[str, int], Mapping[str, Any]] = field(default_factory=dict)

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
        self, fit: Callable[..., RunResult], model_name: str, datasets: Sequence[Dataset], folds: Sequence[Fold]
    ) -> None:
        """Fit *model_name* on every fold of every dataset (original first, then its randomizations).

        :param fit: The fit of the model.
        :param model_name: Name of the model, for the failure records.
        :param datasets: The original dataset, then its randomized copies.
        :param folds: Original folds, each followed by its robustness copies.
        """
        for fold in folds:
            for run_ds in datasets:
                self.fit_run(fit, model_name, run_ds, fold)

    def fit_run(self, fit: Callable[..., RunResult], model_name: str, run_ds: Dataset, fold: Fold) -> None:
        """Load or fit one run: tune an original run and keep its values, or reuse them for a stress-test run.

        The original dataset on an original fold is tuned; every other run (a randomized dataset, or a robustness
        copy of a fold) reuses the best hyperparameters of the original run of its fold.
        """
        identity = self.identity(model_name, fold, run_ds.randomization)
        original = run_ds.randomization is None and "robustness_repeat" not in fold.metadata
        key = (model_name, identity.fold_index)
        result = self.store.load(identity) if self.store is not None else None
        if result is None and not original and key not in self.best_hyperparameters:
            error = OriginalRunFailedError(
                f"the original run of fold {identity.fold_index} failed, so this stress-test run was skipped"
            )
            self.record(error, model_name=model_name, fold=fold, randomized=run_ds.randomization)
        elif result is None:
            hyperparameters = None if original else self.best_hyperparameters[key]
            result = self._fit_new(fit, model_name, run_ds, fold, hyperparameters)
        if result is not None:
            self.runs.append(result)
            if original:
                self.best_hyperparameters[key] = result.best_hyperparameters

    def _fit_new(
        self,
        fit: Callable[..., RunResult],
        model_name: str,
        run_ds: Dataset,
        fold: Fold,
        hyperparameters: Mapping[str, Any] | None,
    ) -> RunResult | None:
        """Fit a run that is not saved yet, and save it when it finishes."""
        result = self.attempt(
            partial(fit, run_ds, fold, hyperparameters=hyperparameters),
            model_name=model_name,
            folds=[fold],
            randomized=run_ds.randomization,
        )
        if result is not None and self.store is not None:
            self.store.save(result)
        return result

    def result(self) -> ExperimentResult:
        """Return the experiment of the finished runs, after recording the failures in the store.

        :returns: The runs and failures as one experiment.
        :raises ExperimentFailedError: If there are failures and no run finished.
        """
        if self.store is not None:
            self.store.finish(self.failures)
        if self.failures and not self.runs:
            raise ExperimentFailedError(self.failures)
        return ExperimentResult(self.runs, self.failures)

    def identity(self, model_name: str, fold: Fold, randomized: tuple[str, str] | None) -> RunIdentity:
        """Return the identity of *model_name*'s run on *fold*, before it is fitted (so without provenance)."""
        return RunIdentity.from_fold(
            model_name, self.dataset_name, canonical(fold.metadata.get("split_mode", "")), fold.metadata, randomized
        )

    def record(
        self,
        error: Exception,
        *,
        model_name: str,
        fold: Fold,
        randomized: tuple[str, str] | None = None,
    ) -> None:
        """Store and log a failure of the run on *fold*."""
        failure = RunFailure.from_exception(error, self.identity(model_name, fold, randomized))
        logger.warning("Run failed, continuing: %s", escape(str(failure)))
        self.failures.append(failure)


def _settings(
    models: Sequence[ModelSpec],
    dataset: Dataset,
    response_transformation: ResponseTransformer | None,
    quality: CurveQuality,
    **arguments: Any,
) -> dict[str, Any]:
    r"""Return the arguments of ``run`` that decide its results, as the JSON that ``metadata.json`` records.

    :param models: The models, recorded by name, configuration and hyperparameters.
    :param dataset: The dataset, recorded by its name and measure only.
    :param response_transformation: Recorded by its class name and ``get_params``.
    :param quality: Recorded by ``to_metadata``.
    :param \**arguments: The other result-deciding arguments of ``run``, with their defaults resolved.
    :returns: Plain JSON values; a value without a JSON form is recorded as its ``str``.
    """
    transformation = None
    if response_transformation is not None:
        transformation = {
            "class": type(response_transformation).__name__,
            "params": response_transformation.get_params(deep=False),
        }
    settings = {
        "models": [
            {
                "name": model.name,
                "config": model.config.model_dump(mode="json"),
                "hyperparameters": thaw_value(model.hyperparameters),
            }
            for model in models
        ],
        "dataset": {"name": dataset.name, "measure": dataset.measure},
        "response_transformation": transformation,
        "quality": quality.to_metadata(),
        **arguments,
    }
    return json.loads(json.dumps(settings, default=str))


def experiment(
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
    validation_ratio: float = 0.1,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> ExperimentResult:
    """Run the full experiment pipeline.

    :param models: Named model specifications to evaluate.
    :param dataset: Dataset object or name to load, or a sequence of them to concatenate into one
        cross-study dataset (see ``drevalpy.data.concat``). A sequence of one is used as it is.
    :param split_mode: Split mode (LPO, LSO, LDO, LTO, LStO).
    :param randomization_modes: Optional randomization modes (SVRC, SVCC, SVRD, SVCD).
    :param randomization_type: How to randomize the view.
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
    :param measure: Response measure to train and score on: ``pEC50`` or any response layer of the dataset, such
        as ``AUC`` (see :meth:`~drevalpy.types.data.dataset.Dataset.with_measure`). ``None``
        keeps the dataset's measure, which is ``LN_IC50`` unless it was set (``pEC50`` for a dataset without an
        ``LN_IC50`` layer).
    :param wandb_project: Weights & Biases project to log to; ``None`` disables tracking.
    :param on_error: ``"record"`` keeps going when a run raises, logs a warning and stores a ``RunFailure`` on
        the result; ``"raise"`` re-raises the first error. ``KeyboardInterrupt`` always propagates.
    :param n_jobs: Number of worker processes a single-drug model fits its per-drug models in, during tuning and in
        every final fit; ``-1`` uses all CPUs. Results are identical for every value, and other models ignore it.
    :param n_splits: Number of CV folds the splitter makes; the k-fold splitters reject fewer than 2. LStO
        ignores it and makes one fold per study.
    :param validation_ratio: Fraction of each fold's training groups the splitter holds out for validation, which
        tuning scores on and models that stop early watch; such a model needs at least two validation groups.
    :param output_dir: Directory to save the results in, laid out as :doc:`/reference/results` describes. Each run
        is saved as soon as it finishes, and a run that is already saved there by a run with the same settings (every
        argument that can change a result; the dataset by its name and measure) is loaded instead of fitted, so a
        killed run resumes. Failures are saved only at the end, and a resumed run retries them. ``None`` saves
        nothing. A directory that holds files but no drevalpy experiment raises ``OutputDirectoryConflictError``.
    :param overwrite: Delete the experiment in *output_dir* first (its ``metadata.json``, ``failures.json`` and model
        directories; other files stay) and start again. Without it, an *output_dir* that holds runs made with other
        settings raises ``OutputDirectoryConflictError``.
    :returns: ExperimentResult grouping all model results, with the recorded failures. If ``on_error="record"``
        and no run produced a result, ``ExperimentFailedError`` is raised instead.
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
    split_mode = canonical(split_mode)
    quality = quality or CurveQuality.default()
    randomization_modes = randomization_modes or []
    hpo_random_state = seed if hpo_random_state is None else hpo_random_state
    randomization_random_state = seed if randomization_random_state is None else randomization_random_state
    store = None
    if output_dir is not None:
        settings = _settings(
            models,
            ds,
            response_transformation,
            quality,
            split_mode=split_mode,
            n_splits=n_splits,
            validation_ratio=validation_ratio,
            randomization_modes=randomization_modes,
            randomization_type=randomization_type,
            randomization_random_state=randomization_random_state,
            robustness_repeats=robustness_repeats,
            hyperparameter_tuning=hyperparameter_tuning,
            hpo_metric=hpo_metric,
            hpo_num_trials=hpo_num_trials,
            hpo_random_state=hpo_random_state,
            seed=seed,
        )
        store = RunStore.open(output_dir, settings, overwrite=overwrite)
    folds = split(
        ds, split_mode, n_splits=n_splits, validation_ratio=validation_ratio, random_state=seed, quality=quality
    )

    if robustness_repeats > 0:
        folds = [s for fold in folds for s in [fold, *robustness(fold, robustness_repeats)]]

    outcomes = _Outcomes(on_error, ds.name, store)

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
            randomization_modes,
            randomization_type=randomization_type,
            random_state=randomization_random_state,
        )
        randomized = outcomes.attempt(build, model_name=model.name, folds=folds) or []
        fit_model = partial(fit, model, response_transformation=response_transformation)
        outcomes.fit_folds(fit_model, model.name, [ds, *randomized], folds)

    return outcomes.result()
