"""``drevalpy single`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


def single_cmd(
    model: Annotated[str, typer.Argument(help="Model name (e.g. ElasticNet, RandomForest).")],
    dataset: Annotated[str, typer.Argument(help="Path to a .h5mu dataset file.")],
    split: Annotated[str, typer.Argument(help="Path to a fold .npz split file.")],
    output: Annotated[str, typer.Argument(help="Output path for the result .npz file.")],
    hpo: Annotated[bool, typer.Option("--hpo/--no-hpo", help="Enable hyperparameter tuning.")] = True,
    hpo_metric: Annotated[str, typer.Option("--hpo-metric", help="Metric to optimize.")] = "RMSE",
    hpo_num_samples: Annotated[int, typer.Option("--hpo-num-samples", help="Number of HPO trials.")] = 16,
    hpo_random_state: Annotated[int, typer.Option("--hpo-random-state", help="HPO random seed.")] = 42,
) -> None:
    """Train a model on one fold, predict on test set, and save results."""
    from drevalpy.models import construct_model
    from drevalpy.single import single as run_single
    from drevalpy.types import SplitMasks
    from drevalpy.types.data.dataset import Dataset

    out = UPath(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model_class = construct_model(model)
    ds = Dataset.load(dataset)
    split_masks = SplitMasks.load(split)

    result = run_single(
        model_class,
        ds,
        split_masks,
        hyperparameter_tuning=hpo,
        hpo_metric=hpo_metric,
        hpo_num_samples=hpo_num_samples,
        hpo_random_state=hpo_random_state,
    )

    result.save(str(out))
    typer.echo(f"Result: {result.model_name} on {result.dataset_name} (fold {result.fold_index}) -> {out}")
