"""``drevalpy evaluate-hpams`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list
from drevalpy.cli_run_cv import run_evaluate_and_find_max


def register(app: typer.Typer) -> None:
    @app.command("evaluate-hpams")
    def evaluate_hpams(
        model_name: Annotated[str, typer.Option("--model_name", help="Model name, used for naming the output file.")],
        split_id: Annotated[str, typer.Option("--split_id", help="Split id, used for naming the output file.")],
        hpam_yamls: Annotated[
            list[str],
            typer.Option(
                "--hpam_yamls",
                help="Paths to hyperparameter configuration yaml files. Pass multiple values separated by spaces.",
            ),
        ],
        pred_datas: Annotated[
            list[str],
            typer.Option(
                "--pred_datas",
                help="Paths to pickled predictions. Pass multiple values separated by spaces.",
            ),
        ],
        optim_metric: Annotated[
            str, typer.Option("--optim_metric", help="Optimization metric, default: RMSE.")
        ] = "RMSE",
    ) -> None:
        """Evaluate predictions and save the best hyperparameter combination."""
        run_evaluate_and_find_max(
            model_name=model_name,
            split_id=split_id,
            hpam_yamls=as_list(hpam_yamls),
            pred_datas=as_list(pred_datas),
            optim_metric=optim_metric,
        )
