"""``drevalpy data align`` command."""

from __future__ import annotations

from typing import Annotated

import typer


def align_dataset(
    input_path: Annotated[str, typer.Argument(help="Path to the existing .h5mu file.")],
    output_path: Annotated[
        str,
        typer.Argument(help="Output .h5mu file path. If omitted, overwrites the input."),
    ] = "",
) -> None:
    """Align an existing .h5mu file to the new featurizer storage conventions.

    Renames old keys to match canonical featurizer storage_key names and
    creates the featurizer_variants uns registry.
    """
    from drevalpy.data.align_mudata import align_mudata

    out = output_path if output_path else None
    align_mudata(input_path, out)
    typer.echo(f"Aligned {input_path}" + (f" -> {output_path}" if output_path else " (in-place)"))
