"""``drevalpy make-final-split-pkls`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from drevalpy.cli.model_testing import run_final_split


def register(app: typer.Typer) -> None:
    """Register the ``make-final-split-pkls`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("make-final-split-pkls")
    def make_final_split_pkls(
        response: Annotated[
            Path,
            typer.Option(
                "--response",
                help="Drug response data, pickled (output of load_response).",
            ),
        ],
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model class name, e.g., RandomForest, SingleDrugRandomForest.",
            ),
        ],
        test_mode: Annotated[
            str,
            typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO). Default: LPO."),
        ] = "LPO",
        val_ratio: Annotated[float, typer.Option("--val_ratio", help="Validation ratio.")] = 0.1,
    ) -> None:
        """Create train/validation/early-stopping pickles for a final production model.

        :param response: Path to the pickled primary response dataset.
        :param model_name: Registered model name used to filter rows by available features.
        :param test_mode: Split label for train/validation partitioning.
        :param val_ratio: Fraction of rows held out for validation.
        """
        run_final_split(
            response=response,
            model_name=model_name,
            test_mode=test_mode,
            val_ratio=val_ratio,
        )
