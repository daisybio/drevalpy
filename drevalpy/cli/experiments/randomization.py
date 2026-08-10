"""``drevalpy experiments randomization`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


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
    ds = Dataset.load(dataset)
    randomized = randomization(model_class, ds, effective_modes, random_state=random_state)

    with Progress() as progress:
        task = progress.add_task("Writing randomized datasets", total=len(randomized))
        for rds in randomized:
            mode_tag, view_tag = rds.randomization or ("unknown", "0")
            out_path = out / f"{mode_tag}:{view_tag}.h5mu"
            rds.mdata.write(str(out_path))
            progress.advance(task)

    typer.echo(f"Wrote {len(randomized)} randomized datasets to {out}")
