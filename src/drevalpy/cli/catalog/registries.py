"""``drevalpy list`` commands over the five component registries.

All five registries expose the same surface (``list``, ``get``, ``metadata``,
``table``), so every command here is a two-line wrapper around :func:`_show`.
"""

from __future__ import annotations

from typing import Annotated

import typer

#: Optional entry name. Given, the command prints that entry's metadata instead
#: of the whole registry table.
EntryName = Annotated[
    str | None,
    typer.Argument(
        metavar="[NAME]",
        help="Show the metadata of a single entry instead of the whole table.",
    ),
]


def _show(registry_name: str, title: str, name: str | None) -> None:
    """Render one registry, or the metadata of a single entry in it.

    Args:
        registry_name: Attribute of :mod:`drevalpy.registry` holding the registry
            module, e.g. ``"predictor"``.
        title: Plural heading for the table, e.g. ``"Predictors"``.
        name: Entry to describe, or ``None`` to render the whole table.

    Raises:
        typer.Exit: With code 1 when ``name`` is not registered. The registry's
            own error message, which lists the registered names, is written to
            stderr first.
    """
    from drevalpy import registry

    from ._render import render_frame, render_mapping

    module = getattr(registry, registry_name)
    if name is None:
        render_frame(
            module.table(),
            title=title,
            empty_hint=f"No {title.lower()} are registered.",
        )
        return
    try:
        metadata = module.metadata(name)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    render_mapping(metadata, title=name)


def list_predictors(name: EntryName = None) -> None:
    """List registered predictors."""
    _show("predictor", "Predictors", name)


def list_cell_line_featurizers(name: EntryName = None) -> None:
    """List registered cell-line featurizers."""
    _show("cell_line_featurizer", "Cell-line featurizers", name)


def list_drug_featurizers(name: EntryName = None) -> None:
    """List registered drug featurizers."""
    _show("drug_featurizer", "Drug featurizers", name)


def list_splitters(name: EntryName = None) -> None:
    """List registered split modes."""
    _show("splitter", "Splitters", name)


def list_visualizations(name: EntryName = None) -> None:
    """List registered visualizations."""
    _show("visualization", "Visualizations", name)
