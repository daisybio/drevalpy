"""``drevalpy run`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


def run_cmd(
    models: Annotated[list[str], typer.Argument(help="Model name(s) to evaluate.")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset name or .h5mu path.")],
    split_mode: Annotated[str, typer.Option("--split-mode", "-s", help="Split mode: LPO, LCO, LDO, LTO.")] = "LPO",
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for results.")] = "results",
    hpo: Annotated[bool, typer.Option("--hpo/--no-hpo", help="Enable hyperparameter tuning.")] = True,
    hpo_metric: Annotated[str, typer.Option("--hpo-metric", help="Metric to optimize.")] = "RMSE",
    hpo_num_samples: Annotated[int, typer.Option("--hpo-num-samples", help="Number of HPO trials.")] = 16,
    hpo_random_state: Annotated[int, typer.Option("--hpo-random-state", help="HPO random seed.")] = 42,
    randomization_mode: Annotated[
        list[str] | None,
        typer.Option("--randomization-mode", "-r", help="Randomization mode(s): SVRC, SVCC, SVRD, SVCD."),
    ] = None,
    robustness_trials: Annotated[
        int, typer.Option("--robustness-trials", help="Number of robustness permutations (0=disabled).")
    ] = 0,
) -> None:
    """Run the full evaluation pipeline."""
    from drevalpy.models import construct_model
    from drevalpy.run import pipeline

    model_classes = [construct_model(m) for m in models]
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = pipeline(
        models=model_classes,
        dataset=dataset,
        split_mode=split_mode,
        randomization_modes=randomization_mode,
        hyperparameter_tuning=hpo,
        hpo_metric=hpo_metric,
        hpo_num_samples=hpo_num_samples,
        hpo_random_state=hpo_random_state,
        robustness_trials=robustness_trials,
    )

    for result in results:
        tag = f"{result.model_name}_fold{result.fold_index}"
        if result.randomization:
            tag += f"_{result.randomization[0]}:{result.randomization[1]}"
        result.save(str(out / f"{tag}.npz"))

    typer.echo(f"Wrote {len(results)} results to {out}")
