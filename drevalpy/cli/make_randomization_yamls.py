"""``drevalpy make-randomization-yamls`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_model_testing import run_randomization_split


def register(app: typer.Typer) -> None:
    @app.command("make-randomization-yamls")
    def make_randomization_yamls(
        model_name: Annotated[str, typer.Option("--model_name", help="Name of the model to use.")],
        randomization_mode: Annotated[str, typer.Option("--randomization_mode", help="Randomization mode to use.")],
    ) -> None:
        """Create randomization test views and save them as yamls."""
        run_randomization_split(model_name=model_name, randomization_mode=randomization_mode)
