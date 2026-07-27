"""``drevalpy make-hpam-yamls`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli.run_cv import run_hpam_split


def register(app: typer.Typer) -> None:
    @app.command("make-hpam-yamls")
    def make_hpam_yamls(
        model_name: Annotated[str, typer.Option("--model_name", help="Model name")],
        hyperparameter_tuning: Annotated[
            bool,
            typer.Option(
                "--hyperparameter_tuning",
                help=(
                    "Deprecated flag kept for nf-core compatibility. Always writes default "
                    "hyperparameters to hpam_0.yaml; Ray/Optuna tuning runs at experiment time."
                ),
            ),
        ] = False,
    ) -> None:
        """Write default hyperparameters to ``hpam_0.yaml`` for nf-core CV subworkflows."""
        run_hpam_split(model_name=model_name, hyperparameter_tuning=hyperparameter_tuning)
