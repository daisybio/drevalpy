"""``drevalpy run`` command."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer
from upath import UPath

from ._quality import MinAbsFoldChange, MinRelevanceScore, NoQualityFilter, quality_from_cli
from ._summary import GroupBy, format_grouped_summary


class ResponseTransformation(StrEnum):
    """Values of ``--response-transformation`` shared by ``run`` and ``single``; ``none`` maps to no transformer."""

    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


def _model_specs(models: list[str], sample: list[str], drug: list[str], predictors: list[str]) -> list:
    """Build the model specs of the MODEL arguments and of the ``-S/-D/-P`` grid, one per name."""
    from drevalpy.models import ModelSpec, combinations

    if not models and not predictors:
        raise typer.BadParameter("Give at least one MODEL or one --predictor/-P.")
    if (sample or drug) and not predictors:
        raise typer.BadParameter("--sample-featurizer/-S and --drug-featurizer/-D need at least one --predictor/-P.")
    specs = [ModelSpec.from_name(m) for m in models]
    if predictors:
        try:
            grid = combinations(sample=sample, drug=drug, predictors=predictors)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        for skipped in grid.skipped:
            typer.echo(f"Skipped {skipped.recipe}: {skipped.reason}")
        specs.extend(grid.specs)
    unique: dict[str, object] = {}
    for spec in specs:
        unique.setdefault(spec.name, spec)
    return list(unique.values())


def _echo_failures(failures) -> None:
    typer.echo(f"{len(failures)} run(s) failed:", err=True)
    for failure in failures:
        typer.echo(f"  {failure}", err=True)


def run_cmd(
    dataset: Annotated[
        list[str],
        typer.Option("--dataset", "-d", help="Dataset name or .h5mu path; repeat to concatenate several studies."),
    ],
    models: Annotated[
        list[str] | None, typer.Argument(help="Zoo name(s) or recipe(s), such as pca[gene_expression]:elasticNet.")
    ] = None,
    split_mode: Annotated[
        str, typer.Option("--split-mode", "-s", help="Split mode: LPO, LSO, LDO, LTO, LStO.")
    ] = "LPO",
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
    group_by: Annotated[
        GroupBy | None,
        typer.Option(
            "--group-by",
            case_sensitive=False,
            help="Also print the metrics computed within each drug or sample instead of pooled over all pairs.",
        ),
    ] = None,
    sample_featurizer: Annotated[
        list[str] | None,
        typer.Option("--sample-featurizer", "-S", help="Sample featurizer recipe for a grid; repeatable."),
    ] = None,
    drug_featurizer: Annotated[
        list[str] | None,
        typer.Option("--drug-featurizer", "-D", help="Drug featurizer recipe for a grid; repeatable."),
    ] = None,
    predictor: Annotated[
        list[str] | None,
        typer.Option("--predictor", "-P", help="Predictor of a grid; repeatable. Every valid combination is run."),
    ] = None,
    no_quality_filter: NoQualityFilter = False,
    min_relevance_score: MinRelevanceScore = None,
    min_abs_fold_change: MinAbsFoldChange = None,
    wandb_project: Annotated[
        str | None, typer.Option("--wandb-project", help="Weights & Biases project to log trials and fits to.")
    ] = None,
) -> None:
    """Run the full evaluation pipeline."""
    from drevalpy._run import run
    from drevalpy.types.results.failure import ExperimentFailedError
    from drevalpy.utils import get_response_transformation

    transformer = (
        None
        if response_transformation is ResponseTransformation.NONE
        else get_response_transformation(response_transformation.value)
    )

    model_specs = _model_specs(models or [], sample_featurizer or [], drug_featurizer or [], predictor or [])
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
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
            quality=quality_from_cli(no_quality_filter, min_relevance_score, min_abs_fold_change),
            wandb_project=wandb_project,
        )
    except ExperimentFailedError as error:
        _echo_failures(error.failures)
        raise typer.Exit(1) from error

    result.save(str(out))
    typer.echo(f"Wrote experiment results to {out}")
    typer.echo(repr(result))
    if group_by is not None:
        typer.echo(format_grouped_summary(result, group_by))
    if result.failures:
        _echo_failures(result.failures)
        raise typer.Exit(1)
