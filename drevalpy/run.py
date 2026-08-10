from drevalpy.models import DRPModel
from drevalpy.types import Dataset
from drevalpy.data import load, split
from drevalpy.experiment.robustness import robustness
from drevalpy.experiment.randomization import randomization

from itertools import product


def pipeline(
    models: list[type[DRPModel]],
    dataset: Dataset | str,
    split_mode: str,
    randomization_modes: list[str] | None = None,
    hyperparameter_tuning: bool = True,
    hpo_metric: str = "RMSE",
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
    robustness_trials: int = 0,
):
    ds = load(dataset) if isinstance(dataset, str) else dataset
    splits = split(ds, split_mode)

    if robustness_trials > 0:
        splits = [s for split in splits for s in robustness(split, robustness_trials)]

    results = []
    for model, split_masks in product(models, splits):
        datasets = [dataset]

        if randomization_modes:
            datasets.extend(randomization(model, dataset, randomization_modes))

        for dataset in datasets:
            result = run(model, dataset, split_masks, hyperparameter_tuning, hpo_metric, hpo_num_samples, hpo_random_state)
            results.append(result)

    return results
