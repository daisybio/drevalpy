"""``drevalpy list`` command group: what is registered in this environment."""

from __future__ import annotations

import typer

from .plugins import list_plugins
from .registries import (
    list_cell_line_featurizers,
    list_drug_featurizers,
    list_predictors,
    list_splitters,
    list_visualizations,
)

list_app = typer.Typer(
    name="list",
    help="List registered components, splitters, visualizations and plugins.",
    no_args_is_help=True,
)

list_app.command("predictors")(list_predictors)
list_app.command("cell-line-featurizers")(list_cell_line_featurizers)
list_app.command("drug-featurizers")(list_drug_featurizers)
list_app.command("splitters")(list_splitters)
list_app.command("visualizations")(list_visualizations)
list_app.command("plugins")(list_plugins)

__all__ = ["list_app"]
