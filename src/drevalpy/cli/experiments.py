"""``drevalpy experiments`` command group: ``robustness`` and ``randomization``."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from drevalpy.errors import InputNotFoundError
from drevalpy.types.enums import RandomizationMode, RandomizationType

experiments_app = typer.Typer(
    name="experiments",
    help="Experiment workflow commands.",
    no_args_is_help=True,
)


def robustness_cmd(
    splits_dir: Annotated[str, typer.Argument(help="Directory containing fold .npz files.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for shuffled split files.")],
    n_repeats: Annotated[int, typer.Option("--n-repeats", "-n", help="Robustness repeats per fold.")] = 5,
) -> None:
    """Generate robustness test splits by shuffling pair ordering.

    Reads each fold .npz from the input directory, produces shuffled variants,
    and writes them to the output directory.
    """
    from rich.progress import Progress

    from drevalpy.experiment._robustness import robustness
    from drevalpy.types import Fold

    inp = UPath(splits_dir)
    out = UPath(output_dir)

    fold_files = sorted(inp.glob("*.npz"))
    if not fold_files:
        raise InputNotFoundError(f"No .npz files found in {inp}")
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    with Progress() as progress:
        task = progress.add_task("Processing folds", total=len(fold_files))
        for fold_file in fold_files:
            fold = Fold.load(str(fold_file))
            variants = robustness(fold, n_repeats)
            for repeat, variant in enumerate(variants):
                out_path = out / f"{fold_file.stem}_repeat_{repeat}.npz"
                variant.save(str(out_path))
                total += 1
            progress.advance(task)

    typer.echo(f"Wrote {total} robustness folds to {out}")


def randomization_cmd(
    model: Annotated[str, typer.Argument(help="Model name (e.g. ElasticNet, SimpleNeuralNetwork).")],
    dataset: Annotated[str, typer.Argument(help="Registered dataset name or path to a .h5mu file.")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for randomized .h5mu files.")],
    modes: Annotated[
        list[RandomizationMode] | None,
        typer.Option("--mode", "-m", case_sensitive=False, help="Randomization mode(s): SVRC, SVCC, SVRD, SVCD."),
    ] = None,
    randomization_type: Annotated[
        RandomizationType,
        typer.Option("--randomization-type", "-t", case_sensitive=False, help="How to replace the view."),
    ] = RandomizationType.PERMUTATION,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
) -> None:
    """Generate randomized datasets for feature importance testing.

    Produces copies of the dataset with views shuffled according to the
    specified randomization modes, based on the model's configured views.
    """
    from rich.progress import Progress

    from drevalpy.data import load
    from drevalpy.experiment._randomization import randomization
    from drevalpy.models import ModelSpec

    effective_modes = [mode.value for mode in modes] if modes else [RandomizationMode.SVRC.value]
    model_spec = ModelSpec.from_name(model)
    ds = load(dataset)
    randomized = randomization(
        model_spec, ds, effective_modes, randomization_type=randomization_type, random_state=seed
    )

    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Writing randomized datasets", total=len(randomized))
        for rds in randomized:
            mode_tag, view_tag = rds.randomization or ("unknown", "0")
            out_path = out / f"{mode_tag}__{view_tag}.h5mu"
            rds.save(str(out_path))
            progress.advance(task)

    typer.echo(f"Wrote {len(randomized)} randomized datasets to {out}")


experiments_app.command("robustness")(robustness_cmd)
experiments_app.command("randomization")(randomization_cmd)
