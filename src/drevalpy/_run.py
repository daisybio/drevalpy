"""Top-level pipeline orchestrating models x folds x randomization."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Literal

from drevalpy._single import single
from drevalpy.data import CurveQuality, load, split
from drevalpy.experiment._randomization import randomization
from drevalpy.experiment._robustness import robustness
from drevalpy.models.spec import ModelSpec
from drevalpy.types import Dataset
from drevalpy.types.results import ExperimentResult, RunResult


def _validate_models(models: list[ModelSpec]) -> None:
    if any(not isinstance(model, ModelSpec) for model in models):
        raise TypeError("run() expects ModelSpec values")
    if len({model.name for model in models}) != len(models):
        raise ValueError("Model names must be unique within an experiment")


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
        cross-study dataset (see ``drevalpy.data.concat``). A sequence is not supported yet.
    :param split_mode: Split mode (LPO, LSO, LDO, LTO).
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
        Not used yet.
    :param wandb_project: Weights & Biases project to log to; ``None`` disables tracking. Not used yet.
    :param on_error: ``"record"`` keeps going when a run raises and stores a ``RunFailure`` on the result;
        ``"raise"`` re-raises the first error. Not used yet.
    :returns: ExperimentResult grouping all model results.
    """
    if isinstance(dataset, Sequence) and not isinstance(dataset, str):
        raise NotImplementedError("W1-P7")
    _validate_models(models)
    ds = load(dataset) if isinstance(dataset, str) else dataset
    folds = split(ds, split_mode, random_state=seed)

    if robustness_trials > 0:
        folds = [s for fold in folds for s in [fold, *robustness(fold, robustness_trials)]]

    randomization_seed = seed if randomization_random_state is None else randomization_random_state
    run_results: list[RunResult] = []

    for model, split_masks in product(models, folds):
        run_datasets: list[Dataset] = [ds]

        if randomization_modes:
            run_datasets.extend(
                randomization(
                    model,
                    ds,
                    randomization_modes,
                    randomization_type=randomization_type,
                    random_state=randomization_seed,
                )
            )

        for run_ds in run_datasets:
            result = single(
                model,
                run_ds,
                split_masks,
                hyperparameter_tuning=hyperparameter_tuning,
                hpo_metric=hpo_metric,
                hpo_num_trials=hpo_num_trials,
                hpo_random_state=hpo_random_state,
                seed=seed,
                response_transformation=response_transformation,
            )
            run_results.append(result)

    return ExperimentResult(run_results)
