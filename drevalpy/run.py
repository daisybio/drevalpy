"""Top-level pipeline orchestrating models x folds x randomization."""

from __future__ import annotations

from itertools import product

from drevalpy.data import load, split
from drevalpy.experiment.randomization import randomization
from drevalpy.experiment.robustness import robustness
from drevalpy.models.drp_model import DRPModel
from drevalpy.single import single
from drevalpy.types import Dataset
from drevalpy.types.results import ExperimentResult, RunResult


def run(
    models: list[type[DRPModel]],
    dataset: Dataset | str,
    split_mode: str,
    randomization_modes: list[str] | None = None,
    hyperparameter_tuning: bool = True,
    hpo_metric: str = "RMSE",
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
    robustness_trials: int = 0,
    precomputed_only: bool = False,
) -> ExperimentResult:
    """Run the full experiment pipeline.

    :param models: Model classes to evaluate.
    :param dataset: Dataset object or name to load.
    :param split_mode: Split mode (LPO, LCO, LDO, LTO).
    :param randomization_modes: Optional randomization modes (SVRC, SVCC, SVRD, SVCD).
    :param hyperparameter_tuning: Whether to run HPO.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_samples: Number of HPO trials.
    :param hpo_random_state: Random seed for HPO.
    :param robustness_trials: Number of robustness permutations (0 = disabled).
    :param precomputed_only: Restrict HPO to pre-computed featurizer variants.
    :returns: ExperimentResult grouping all model results.
    """
    ds = load(dataset) if isinstance(dataset, str) else dataset
    folds = split(ds, split_mode)

    if robustness_trials > 0:
        folds = [s for fold in folds for s in robustness(fold, robustness_trials)]

    run_results: list[RunResult] = []

    for model, split_masks in product(models, folds):
        run_datasets: list[Dataset] = [ds]

        if randomization_modes:
            run_datasets.extend(randomization(model, ds, randomization_modes))

        for run_ds in run_datasets:
            result = single(
                model,
                run_ds,
                split_masks,
                hyperparameter_tuning=hyperparameter_tuning,
                hpo_metric=hpo_metric,
                hpo_num_samples=hpo_num_samples,
                hpo_random_state=hpo_random_state,
                precomputed_only=precomputed_only,
            )
            run_results.append(result)

    return ExperimentResult(run_results)
