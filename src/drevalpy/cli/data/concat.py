"""``drevalpy data concat`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


def concat_datasets(
    datasets: Annotated[
        list[str],
        typer.Argument(help="Two or more registered dataset names or .h5mu paths, one per study."),
    ],
    output: Annotated[str, typer.Argument(help="Output .h5mu file path.")],
    join: Annotated[
        str,
        typer.Option(
            "--join",
            help="'exact' needs identical feature names in every shared view; 'inner' keeps the shared features.",
        ),
    ] = "exact",
    name: Annotated[
        str | None,
        typer.Option("--name", help="Name of the result. Defaults to the dataset names joined by '+'."),
    ] = None,
) -> None:
    """Concatenate datasets of different studies into one dataset.

    Each row becomes ``<study>::<sample_id>``, ``obs`` gains ``study`` and
    ``source_sample_id``, views are stacked by rows and drug columns are united.
    Run any split mode on the result, such as ``LStO`` to train on one study and
    test on another.
    """
    from drevalpy.data import concat, load

    path = UPath(output)
    try:
        combined = concat([load(item) for item in datasets], join=join, name=name)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    combined.save(str(path))
    typer.echo(f"Wrote {combined.name} to {path}")
