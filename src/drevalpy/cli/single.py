"""``drevalpy single`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from drevalpy.evaluation import Metric

from ._options import (
    Hpo,
    HpoMetric,
    HpoNumTrials,
    HpoRandomState,
    NJobs,
    ResponseTransformation,
    ResponseTransformationOption,
    WandbProject,
    resolve_response_transformation,
)


def single_cmd(
    model: Annotated[str, typer.Argument(help="Model name (e.g. ElasticNet, RandomForest).")],
    dataset: Annotated[str, typer.Argument(help="Path to a .h5mu dataset file.")],
    split: Annotated[str, typer.Argument(help="Path to a fold .npz split file.")],
    output: Annotated[str, typer.Argument(help="Output path for the result .npz file.")],
    hpo: Hpo = True,
    hpo_metric: HpoMetric = Metric.RMSE,
    hpo_num_trials: HpoNumTrials = 16,
    seed: Annotated[int, typer.Option("--seed", help="Root seed for HPO and model fits.")] = 42,
    hpo_random_state: HpoRandomState = None,
    response_transformation: ResponseTransformationOption = ResponseTransformation.NONE,
    split_mode: Annotated[str, typer.Option("--split-mode", help="Split mode fallback (e.g. LPO, LSO, LDO).")] = "LPO",
    wandb_project: WandbProject = None,
    n_jobs: NJobs = 1,
) -> None:
    """Train a model on one fold, predict on test set, and save results."""
    from drevalpy._single import single as run_single
    from drevalpy.models import ModelSpec
    from drevalpy.registry.splitter import splitter_registry
    from drevalpy.types import SplitMasks
    from drevalpy.types.data.dataset import Dataset

    model_spec = ModelSpec.from_name(model)
    transformer = resolve_response_transformation(response_transformation)
    ds = Dataset.load(dataset)
    split_masks = SplitMasks.load(split)
    split_masks.metadata["split_mode"] = splitter_registry.canonical(split_masks.metadata.get("split_mode", split_mode))

    out = UPath(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    result = run_single(
        model_spec,
        ds,
        split_masks,
        hyperparameter_tuning=hpo,
        response_transformation=transformer,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        seed=seed,
        wandb_project=wandb_project,
        n_jobs=n_jobs,
    )

    result.save(str(out))
    typer.echo(f"Result: {result.model_name} on {result.dataset_name} (fold {result.fold_index}) -> {out}")
