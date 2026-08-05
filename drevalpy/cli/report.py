"""``drevalpy report`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.visualization.create_report import run_report


def register(app: typer.Typer) -> None:
    """Register the ``report`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("report")
    def report(
        run_id: Annotated[str, typer.Option("--run_id", help="Run ID for the current execution")],
        dataset_name: Annotated[
            str, typer.Option("--dataset_name", help="Name of the dataset for which to render the result file")
        ],
        path_data: Annotated[str, typer.Option("--path_data", help="Path to the data")] = "data",
        result_path: Annotated[str, typer.Option("--result_path", help="Path to the results")] = "results",
    ) -> None:
        """Generate reports from evaluation results.

        :param run_id: Experiment run identifier.
        :param dataset_name: Dataset name used to locate result files.
        :param path_data: Root data directory for report assets.
        :param result_path: Directory containing experiment outputs.
        """
        run_report(run_id=run_id, dataset=dataset_name, path_data=path_data, result_path=result_path)
