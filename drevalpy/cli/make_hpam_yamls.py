"""``drevalpy make-hpam-yamls`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_run_cv import run_hpam_split


def register(app: typer.Typer) -> None:
    @app.command("make-hpam-yamls")
    def make_hpam_yamls(
        model_name: Annotated[str, typer.Option("--model_name", help="Model name")],
        hyperparameter_tuning: Annotated[
            bool,
            typer.Option(
                "--hyperparameter_tuning",
                help="If set, hyperparameter tuning is performed, otherwise only the first combination is used",
            ),
        ] = False,
    ) -> None:
        """Create one yaml for each unique hyperparameter combination (hpam_0.yaml, hpam_1.yaml, ...)."""
        run_hpam_split(model_name=model_name, hyperparameter_tuning=hyperparameter_tuning)
