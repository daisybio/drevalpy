"""``drevalpy train-cv`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli.run_cv import run_train_and_predict_cv


def register(app: typer.Typer) -> None:
    """Register the ``train-cv`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("train-cv")
    def train_cv(
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
            ),
        ],
        hyperparameters: Annotated[
            str,
            typer.Option(
                "--hyperparameters",
                help="Path to the yaml file containing the hyperparameter configuration for this run.",
            ),
        ],
        cv_data: Annotated[str, typer.Option("--cv_data", help="Path to the pickled cv data split.")],
        path_data: Annotated[str, typer.Option("--path_data", help="Data directory path, default: data.")] = "data",
        test_mode: Annotated[
            str,
            typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO), default: LPO."),
        ] = "LPO",
        response_transformation: Annotated[
            str,
            typer.Option(
                "--response_transformation",
                help="Response transformation to apply to the dataset, default: None.",
            ),
        ] = "None",
        model_checkpoint_dir: Annotated[
            str,
            typer.Option(
                "--model_checkpoint_dir",
                help="model checkpoint directory, if not provided: temporary directory is used",
            ),
        ] = "TEMPORARY",
    ) -> None:
        """Train on a CV split and save validation predictions as pickle.

        :param model_name: Registered model name (optionally ``Model.drug`` for single-drug).
        :param hyperparameters: Path to a YAML hyperparameter file.
        :param cv_data: Path to a pickled CV split artifact.
        :param path_data: Root data directory passed to feature loaders.
        :param test_mode: Split label used when naming outputs.
        :param response_transformation: Sklearn response transform name.
        :param model_checkpoint_dir: Directory for model checkpoints.
        """
        run_train_and_predict_cv(
            model_name=model_name,
            path_data=path_data,
            test_mode=test_mode,
            hyperparameters=hyperparameters,
            cv_data=cv_data,
            response_transformation=response_transformation,
            model_checkpoint_dir=model_checkpoint_dir,
        )
