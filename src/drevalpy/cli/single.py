"""``drevalpy single`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from .run import ResponseTransformation


def single_cmd(
    model: Annotated[str, typer.Argument(help="Model name (e.g. ElasticNet, RandomForest).")],
    dataset: Annotated[str, typer.Argument(help="Path to a .h5mu dataset file.")],
    split: Annotated[str, typer.Argument(help="Path to a fold .npz split file.")],
    output: Annotated[str, typer.Argument(help="Output path for the result .npz file.")],
    hpo: Annotated[bool, typer.Option("--hpo/--no-hpo", help="Enable hyperparameter tuning.")] = True,
    hpo_metric: Annotated[str, typer.Option("--hpo-metric", help="Metric to optimize.")] = "RMSE",
    hpo_num_trials: Annotated[int, typer.Option("--hpo-trials", help="Number of HPO trials.")] = 16,
    seed: Annotated[int, typer.Option("--seed", help="Root seed for HPO and model fits.")] = 42,
    hpo_random_state: Annotated[
        int | None, typer.Option("--hpo-random-state", help="HPO sampler seed (default: --seed).")
    ] = None,
    response_transformation: Annotated[
        ResponseTransformation,
        typer.Option(
            "--response-transformation",
            case_sensitive=False,
            help="Response scaling fitted on the training scope.",
        ),
    ] = ResponseTransformation.NONE,
    split_mode: Annotated[str, typer.Option("--split-mode", help="Split mode fallback (e.g. LPO, LSO, LDO).")] = "LPO",
    wandb_project: Annotated[
        str | None,
        typer.Option("--wandb-project", help="Weights & Biases project to log the trials and the final fit to."),
    ] = None,
) -> None:
    """Train a model on one fold, predict on test set, and save results."""
    from drevalpy._single import single as run_single
    from drevalpy.models import ModelSpec
    from drevalpy.registry.splitter import splitter_registry
    from drevalpy.types import SplitMasks
    from drevalpy.types.data.dataset import Dataset
    from drevalpy.utils import get_response_transformation

    transformer = (
        None
        if response_transformation is ResponseTransformation.NONE
        else get_response_transformation(response_transformation.value)
    )

    out = UPath(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model_spec = ModelSpec.from_name(model)
    ds = Dataset.load(dataset)
    split_masks = SplitMasks.load(split)
    split_masks.metadata["split_mode"] = splitter_registry.canonical(split_masks.metadata.get("split_mode", split_mode))

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
    )

    result.save(str(out))
    typer.echo(f"Result: {result.model_name} on {result.dataset_name} (fold {result.fold_index}) -> {out}")
