"""``drevalpy test-cv`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list
from drevalpy.cli_model_testing import run_train_and_predict_final


def register(app: typer.Typer) -> None:
    @app.command("test-cv")
    def test_cv(
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model name for global models, <Model name>.<Drug name> for single-drug models.",
            ),
        ],
        split_id: Annotated[str, typer.Option("--split_id", help="Split id.")],
        split_dataset_path: Annotated[
            str, typer.Option("--split_dataset_path", help="Path to the pickled CV split dataset.")
        ],
        hyperparameters_path: Annotated[
            str,
            typer.Option("--hyperparameters_path", help="Path to yaml file containing the optimal hyperparameters."),
        ],
        path_data: Annotated[str, typer.Option("--path_data", help="Path to data. Default: data")] = "data",
        mode: Annotated[
            str, typer.Option("--mode", help="Mode: full, randomization, or robustness. Default: full.")
        ] = "full",
        response_transformation: Annotated[
            str, typer.Option("--response_transformation", help="Response transformation. Default: None.")
        ] = "None",
        test_mode: Annotated[
            str, typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO). Default: LPO.")
        ] = "LPO",
        randomization_views_path: Annotated[
            str | None,
            typer.Option(
                "--randomization_views_path",
                help="Path to the yaml file containing the randomization configuration "
                "(only relevant if mode=randomization).",
            ),
        ] = None,
        randomization_type: Annotated[
            str,
            typer.Option(
                "--randomization_type",
                help="Randomization type (permutation, invariant). Default: permutation. "
                "Only relevant if mode=randomization.",
            ),
        ] = "permutation",
        robustness_trial: Annotated[
            int | None,
            typer.Option("--robustness_trial", help="Robustness trial index. Only relevant if mode=robustness."),
        ] = None,
        cross_study_datasets: Annotated[
            list[str] | None,
            typer.Option("--cross_study_datasets", help="Paths to pickled cross study datasets (space-separated)."),
        ] = None,
        model_checkpoint_dir: Annotated[
            str,
            typer.Option(
                "--model_checkpoint_dir",
                help="model checkpoint directory, if not provided: temporary directory is used",
            ),
        ] = "TEMPORARY",
    ) -> None:
        """Train and predict on the CV test set (full, randomization, or robustness mode)."""
        run_train_and_predict_final(
            mode=mode,
            model_name=model_name,
            split_id=split_id,
            split_dataset_path=split_dataset_path,
            hyperparameters_path=hyperparameters_path,
            response_transformation=response_transformation,
            test_mode=test_mode,
            path_data=path_data,
            randomization_views_path=randomization_views_path,
            randomization_type=randomization_type,
            robustness_trial=robustness_trial,
            cross_study_datasets=as_list(cross_study_datasets) if cross_study_datasets else None,
            model_checkpoint_dir=model_checkpoint_dir,
        )
