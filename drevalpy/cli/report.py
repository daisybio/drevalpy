"""``drevalpy report`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.visualization.create_report import run_report


def register(app: typer.Typer) -> None:
    @app.command("report")
    def report(
        run_id: Annotated[str, typer.Option("--run_id", help="Run ID for the current execution")],
        dataset: Annotated[str, typer.Option("--dataset", help="Dataset name for which to render the result file")],
        path_data: Annotated[str, typer.Option("--path_data", help="Path to the data")] = "data",
        result_path: Annotated[str, typer.Option("--result_path", help="Path to the results")] = "results",
    ) -> None:
        """Generate reports from evaluation results."""
        run_report(run_id=run_id, dataset=dataset, path_data=path_data, result_path=result_path)
