"""``drevalpy make-cv-pkls`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_run_cv import run_cv_split


def register(app: typer.Typer) -> None:
    @app.command("make-cv-pkls")
    def make_cv_pkls(
        response: Annotated[str, typer.Option("--response", help="Path to the pickled response data file.")],
        n_cv_splits: Annotated[int, typer.Option("--n_cv_splits", help="Number of CV splits")],
        test_mode: Annotated[
            str, typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO), default: LPO.")
        ] = "LPO",
        validation_ratio: Annotated[
            float, typer.Option("--validation_ratio", help="Ratio of validation data, default: 0.1")
        ] = 0.1,
        seed: Annotated[int, typer.Option("--seed", help="Random seed for splitting the data, default: 42.")] = 42,
    ) -> None:
        """Split data into CV splits: split_0.pkl, split_1.pkl, ..."""
        run_cv_split(
            response=response,
            n_cv_splits=n_cv_splits,
            test_mode=test_mode,
            validation_ratio=validation_ratio,
            seed=seed,
        )
