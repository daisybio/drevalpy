"""``drevalpy run`` command."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer
from upath import UPath


class ResponseTransformation(StrEnum):
    """Values of ``--response-transformation`` shared by ``run`` and ``single``; ``none`` maps to no transformer."""

    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


def run_cmd(
    models: Annotated[list[str], typer.Argument(help="Model name(s) to evaluate.")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset name or .h5mu path.")],
    split_mode: Annotated[str, typer.Option("--split-mode", "-s", help="Split mode: LPO, LSO, LDO, LTO.")] = "LPO",
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for results.")] = "results",
    hpo: Annotated[bool, typer.Option("--hpo/--no-hpo", help="Enable hyperparameter tuning.")] = True,
    hpo_metric: Annotated[str, typer.Option("--hpo-metric", help="Metric to optimize.")] = "RMSE",
    hpo_num_trials: Annotated[int, typer.Option("--hpo-trials", help="Number of HPO trials.")] = 16,
    seed: Annotated[
        int, typer.Option("--seed", help="Root seed for splits, HPO, feature randomization, and model fits.")
    ] = 42,
    hpo_random_state: Annotated[
        int | None, typer.Option("--hpo-random-state", help="HPO sampler seed (default: --seed).")
    ] = None,
    randomization_mode: Annotated[
        list[str] | None,
        typer.Option("--randomization-mode", "-r", help="Randomization mode(s): SVRC, SVCC, SVRD, SVCD."),
    ] = None,
    randomization_type: Annotated[
        str, typer.Option("--randomization-type", help="Randomization type: permutation or invariant.")
    ] = "permutation",
    randomization_random_state: Annotated[
        int | None,
        typer.Option("--randomization-random-state", help="Root seed for feature randomization (default: --seed)."),
    ] = None,
    robustness_trials: Annotated[
        int, typer.Option("--robustness-trials", help="Number of robustness permutations (0=disabled).")
    ] = 0,
    response_transformation: Annotated[
        ResponseTransformation,
        typer.Option(
            "--response-transformation",
            case_sensitive=False,
            help="Response scaling fitted on the training scope.",
        ),
    ] = ResponseTransformation.NONE,
) -> None:
    """Run the full evaluation pipeline."""
    from drevalpy._run import run
    from drevalpy.models import ModelSpec
    from drevalpy.utils import get_response_transformation

    transformer = (
        None
        if response_transformation is ResponseTransformation.NONE
        else get_response_transformation(response_transformation.value)
    )

    model_specs = [ModelSpec.from_name(m) for m in models]
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = run(
        models=model_specs,
        dataset=dataset,
        split_mode=split_mode,
        randomization_modes=randomization_mode,
        randomization_type=randomization_type,
        seed=seed,
        randomization_random_state=randomization_random_state,
        hyperparameter_tuning=hpo,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        robustness_trials=robustness_trials,
        response_transformation=transformer,
    )

    result.save(str(out))
    typer.echo(f"Wrote experiment results to {out}")
    typer.echo(repr(result))
