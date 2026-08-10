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

    total = 0
    with typer.progressbar(fold_files, label="Processing folds") as progress:
        for fold_file in progress:
            fold = SplitMasks.load(str(fold_file))
            variants = robustness(fold, n_permutations)
            for trial, variant in enumerate(variants):
                out_path = out / f"{fold_file.stem}_trial_{trial}.npz"
                variant.save(str(out_path))
                total += 1

    typer.echo(f"Wrote {total} robustness splits to {out}")
