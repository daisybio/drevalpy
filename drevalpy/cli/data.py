"""``drevalpy data`` command group."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

data_app = typer.Typer(
    name="data",
    help="Data management commands.",
    no_args_is_help=True,
)


@data_app.command("load")
def load_dataset(
    name: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output: Annotated[str, typer.Argument(help="Output .h5mu file path.")],
) -> None:
    """Load a dataset and write it to an output file.

    Resolves the dataset by name (downloading if needed) and writes it as .h5mu.
    """
    from drevalpy.data import load

    path = UPath(output)
    dataset = load(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.mdata.write(str(path))
    typer.echo(f"Wrote {dataset.name} to {path}")


@data_app.command("split")
def split_dataset(
    dataset: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for split .npz files.")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Split mode: LPO, LCO, LDO, or LTO.")] = "LPO",
    n_splits: Annotated[int, typer.Option("--n-splits", "-n", help="Number of CV folds.")] = 5,
    validation_ratio: Annotated[
        float, typer.Option("--validation-ratio", help="Fraction of training data for validation.")
    ] = 0.1,
    random_state: Annotated[int, typer.Option("--random-state", help="Random seed.")] = 42,
) -> None:
    """Split a dataset into cross-validation folds.

    Writes one .npz file per fold to the output directory.
    """
    from drevalpy.data import Dataset, split

    dataset = Dataset.from_file(dataset)

    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    folds = split(dataset, mode=mode, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state)

    from rich.progress import Progress

    with Progress() as progress:
        task = progress.add_task("Writing folds", total=len(folds))
        for i, fold in enumerate(folds):
            fold_path = out / f"fold_{i}.npz"
            fold.save(str(fold_path))
            progress.advance(task)

    typer.echo(f"Wrote {len(folds)} folds to {out}")
