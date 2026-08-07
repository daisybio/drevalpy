"""``drevalpy make-pipeline-report`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list
from drevalpy.visualization.create_report import run_pipeline_report


def register(app: typer.Typer) -> None:
    """Register the ``make-pipeline-report`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("make-pipeline-report")
    def make_pipeline_report(
        test_modes: Annotated[
            list[str],
            typer.Option(
                "--test_modes",
                help="LPO, LDO, LCO, or LTO. Pass multiple values separated by spaces.",
            ),
        ],
        eval_results: Annotated[str, typer.Option("--eval_results", help="Path to the evaluation results.")],
        eval_results_per_drug: Annotated[
            str, typer.Option("--eval_results_per_drug", help="Path to the evaluation results per drug.")
        ],
        eval_results_per_cl: Annotated[
            str, typer.Option("--eval_results_per_cl", help="Path to the evaluation results per cell line.")
        ],
        true_vs_predicted: Annotated[
            str, typer.Option("--true_vs_predicted", help="Path to the true vs predicted results.")
        ],
        path_data: Annotated[Path, typer.Option("--path_data", help="Path to the data.")],
    ) -> None:
        """Make the HTML report for the pipeline.

        :param test_modes: Split labels to include in the report.
        :param eval_results: Path to aggregated evaluation results.
        :param eval_results_per_drug: Path to per-drug evaluation results.
        :param eval_results_per_cl: Path to per-cell-line evaluation results.
        :param true_vs_predicted: Path to true-versus-predicted scatter data.
        :param path_data: Root data directory for report assets.
        """
        run_pipeline_report(
            test_modes=as_list(test_modes),
            eval_results=eval_results,
            eval_results_per_drug=eval_results_per_drug,
            eval_results_per_cl=eval_results_per_cl,
            true_vs_predicted=true_vs_predicted,
            path_data=path_data,
        )
