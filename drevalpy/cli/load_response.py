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
        clean_min_responders: Annotated[
            int | None,
            typer.Option(
                "--clean_min_responders",
                help="Keep only drugs with at least this many reproducible responder curves. "
                "Requires curve-curated data. Set at most one of the two clean options.",
            ),
        ] = None,
        clean_min_responder_frac: Annotated[
            float | None,
            typer.Option(
                "--clean_min_responder_frac",
                help="Fraction-based alternative to --clean_min_responders (in (0, 1]).",
            ),
        ] = None,
    ) -> None:
        """Load drug response data for drug response prediction as pickle."""
        run_load_response(
            response_dataset=response_dataset,
            cross_study_dataset=cross_study_dataset,
            measure=measure,
            clean_min_responders=clean_min_responders,
            clean_min_responder_frac=clean_min_responder_frac,
        )
