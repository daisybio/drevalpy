"""``drevalpy make-randomization-yamls`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli.model_testing import run_randomization_split


def register(app: typer.Typer) -> None:
    """Register the ``make-randomization-yamls`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("make-randomization-yamls")
    def make_randomization_yamls(
        model_name: Annotated[str, typer.Option("--model_name", help="Name of the model to use.")],
        randomization_mode: Annotated[str, typer.Option("--randomization_mode", help="Randomization mode to use.")],
    ) -> None:
        """Create randomization test views and save them as yamls.

        :param model_name: Registered model name.
        :param randomization_mode: Randomization mode (SVCC, SVRC, SVCD, or SVRD).
        """
        run_randomization_split(model_name=model_name, randomization_mode=randomization_mode)
