"""``drevalpy data concat`` command."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer
from upath import UPath


class Join(StrEnum):
    """Values of ``--join``: how views that disagree on feature names are combined."""

    EXACT = "exact"
    INNER = "inner"


def concat_datasets(
    datasets: Annotated[
        list[str],
        typer.Argument(help="Two or more registered dataset names or .h5mu paths, one per study."),
    ],
    output: Annotated[str, typer.Argument(help="Output .h5mu file path.")],
    join: Annotated[
        Join,
        typer.Option(
            "--join",
            case_sensitive=False,
            help="'exact' needs identical feature names in every shared view; 'inner' keeps the shared features.",
        ),
    ] = Join.EXACT,
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

    combined = concat([load(item) for item in datasets], join=join.value, name=name)
    path = UPath(output)
    combined.save(str(path))
    typer.echo(f"Wrote {combined.name} to {path}")
