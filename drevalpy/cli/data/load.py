"""``drevalpy data load`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


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
