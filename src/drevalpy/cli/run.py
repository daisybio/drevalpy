"""``drevalpy run`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


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
) -> None:
    """Run the full evaluation pipeline."""
    from drevalpy._run import run
    from drevalpy.models import ModelSpec

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
    )

    result.save(str(out))
    typer.echo(f"Wrote experiment results to {out}")
    typer.echo(repr(result))
