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
