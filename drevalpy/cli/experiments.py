"""``drevalpy experiments`` command group."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

experiments_app = typer.Typer(
    name="experiments",
    help="Experiment workflow commands.",
    no_args_is_help=True,
)


@experiments_app.command("robustness")
def robustness_cmd(
    splits_dir: Annotated[str, typer.Argument(help="Directory containing fold .npz files.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for shuffled split files.")],
    n_permutations: Annotated[
        int, typer.Option("--n-permutations", "-n", help="Number of shuffled variants per fold.")
    ] = 5,
) -> None:
    """Generate robustness test splits by shuffling pair ordering.

    Reads each fold .npz from the input directory, produces shuffled variants,
    and writes them to the output directory.
    """
    from drevalpy.experiment.robustness import robustness
    from drevalpy.types import SplitMasks

    inp = UPath(splits_dir)
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fold_files = sorted(inp.glob("*.npz"))
    if not fold_files:
        typer.echo(f"No .npz files found in {inp}", err=True)
        raise typer.Exit(code=1)

    from rich.progress import Progress

    total = 0
    with Progress() as progress:
        task = progress.add_task("Processing folds", total=len(fold_files))
        for fold_file in fold_files:
            fold = SplitMasks.load(str(fold_file))
            variants = robustness(fold, n_permutations)
            for trial, variant in enumerate(variants):
                out_path = out / f"{fold_file.stem}_trial_{trial}.npz"
                variant.save(str(out_path))
                total += 1
            progress.advance(task)

    typer.echo(f"Wrote {total} robustness splits to {out}")


@experiments_app.command("randomization")
def randomization_cmd(
    model: Annotated[str, typer.Argument(help="Model name (e.g. ElasticNet, SimpleNeuralNetwork).")],
    dataset: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for randomized .h5mu files.")],
    modes: Annotated[
        list[str] | None,
        typer.Option("--mode", "-m", help="Randomization mode(s): SVRC, SVCC, SVRD, SVCD."),
    ] = None,
    random_state: Annotated[int, typer.Option("--random-state", help="Random seed.")] = 42,
) -> None:
    """Generate randomized datasets for feature importance testing.

    Produces copies of the dataset with views shuffled according to the
    specified randomization modes, based on the model's configured views.
    """
    from rich.progress import Progress

    from drevalpy.data import Dataset
    from drevalpy.experiment.randomization import randomization
    from drevalpy.models import construct_model

    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    effective_modes = modes if modes else ["SVRC"]
    model_class = construct_model(model)
    ds = Dataset.from_file(dataset)
    randomized = randomization(model_class, ds, effective_modes, random_state=random_state)

    with Progress() as progress:
        task = progress.add_task("Writing randomized datasets", total=len(randomized))
        for i, rds in enumerate(randomized):
            mode_tag, view_tag = rds.randomization or ("unknown", str(i))
            out_path = out / f"{mode_tag}:{view_tag}.h5mu"
            rds.mdata.write(str(out_path))
            progress.advance(task)

    typer.echo(f"Wrote {len(randomized)} randomized datasets to {out}")


@experiments_app.command("run")
def run_cmd(
    model: Annotated[str, typer.Argument(help="Model name (e.g. ElasticNet, RandomForest).")],
    dataset: Annotated[str, typer.Argument(help="Path to a .h5mu dataset file.")],
    split: Annotated[str, typer.Argument(help="Path to a fold .npz split file.")],
    output: Annotated[str, typer.Argument(help="Output path for the result .npz file.")],
    hpo: Annotated[bool, typer.Option("--hpo/--no-hpo", help="Enable hyperparameter tuning.")] = True,
    hpo_metric: Annotated[str, typer.Option("--hpo-metric", help="Metric to optimize.")] = "RMSE",
    hpo_num_samples: Annotated[int, typer.Option("--hpo-num-samples", help="Number of HPO trials.")] = 16,
    hpo_random_state: Annotated[int, typer.Option("--hpo-random-state", help="HPO random seed.")] = 42,
) -> None:
    """Train a model on one fold, predict on test set, and save results.

    Runs the full single-fold pipeline: optional HPO, final training,
    prediction, and metric computation.
    """
    from drevalpy.experiment.run import run
    from drevalpy.models import construct_model
    from drevalpy.types import SplitMasks
    from drevalpy.types.dataset import Dataset

    out = UPath(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model_class = construct_model(model)
    ds = Dataset.from_file(dataset)
    split_masks = SplitMasks.load(split)

    result = run(
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
