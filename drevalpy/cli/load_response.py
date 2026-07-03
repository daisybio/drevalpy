"""``drevalpy load-response`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_run_cv import run_load_response


def register(app: typer.Typer) -> None:
    @app.command("load-response")
    def load_response(
        response_dataset: Annotated[
            str,
            typer.Option("--response_dataset", help="Path to the drug response file dataset_name.csv."),
        ],
        cross_study_dataset: Annotated[
            bool,
            typer.Option("--cross_study_dataset", help="Whether to load cross-study datasets, default: False."),
        ] = False,
        measure: Annotated[
            str,
            typer.Option(
                "--measure",
                help="Name of the column in the dataset containing the drug response measures, "
                "default: LN_IC50_curvecurator.",
            ),
        ] = "LN_IC50_curvecurator",
    ) -> None:
        """Load drug response data for drug response prediction as pickle."""
        run_load_response(
            response_dataset=response_dataset,
            cross_study_dataset=cross_study_dataset,
            measure=measure,
        )
