"""``drevalpy make-cv-pkls`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli.run_cv import run_cv_split


def register(app: typer.Typer) -> None:
    """Register the ``make-cv-pkls`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("make-cv-pkls")
    def make_cv_pkls(
        response: Annotated[
            str,
            typer.Option("--response", help="Path to the pickled response data file."),
        ],
        n_cv_splits: Annotated[int, typer.Option("--n_cv_splits", help="Number of CV splits")],
        test_mode: Annotated[
            str,
            typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO), default: LPO."),
        ] = "LPO",
        validation_ratio: Annotated[
            float,
            typer.Option("--validation_ratio", help="Ratio of validation data, default: 0.1"),
        ] = 0.1,
        seed: Annotated[
            int,
            typer.Option("--seed", help="Random seed for splitting the data, default: 42."),
        ] = 42,
        custom_splitter_path: Annotated[
            str | None,
            typer.Option(
                "--custom_splitter_path",
                help="Path to a Python script defining create_splits(response_data, params).",
            ),
        ] = None,
    ) -> None:
        """Split data into CV splits: split_0.pkl, split_1.pkl, ...

        :param response: Path to the pickled primary response dataset.
        :param n_cv_splits: Number of cross-validation folds to create.
        :param test_mode: Split label (LPO, LCO, LTO, or LDO).
        :param validation_ratio: Fraction of training rows held out for validation.
        :param seed: Random seed for splitting.
        :param custom_splitter_path: Optional external Python split script path.
        """
        run_cv_split(
            response=response,
            n_cv_splits=n_cv_splits,
            test_mode=test_mode,
            validation_ratio=validation_ratio,
            seed=seed,
            custom_splitter_path=custom_splitter_path,
        )
