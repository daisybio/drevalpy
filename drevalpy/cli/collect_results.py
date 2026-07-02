"""``drevalpy collect-results`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list
from drevalpy.cli_model_testing import run_collect_results


def register(app: typer.Typer) -> None:
    @app.command("collect-results")
    def collect_results(
        outfiles: Annotated[
            list[str],
            typer.Option(
                "--outfiles",
                help="Output files containing results (evaluation_results*csv + true_vs_pred.csv). "
                "Pass multiple values separated by spaces.",
            ),
        ],
        path_data: Annotated[str, typer.Option("--path_data", help="Data directory path. Default: data.")] = "data",
    ) -> None:
        """Collect results and write to single files."""
        run_collect_results(outfiles=as_list(outfiles), path_data=path_data)
