"""``drevalpy run`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from upath import UPath

from drevalpy.evaluation import Metric
from drevalpy.types.enums.randomization import RandomizationType

from ._options import (
    GroupByOption,
    Hpo,
    HpoMetric,
    HpoNumTrials,
    HpoRandomState,
    NJobs,
    RandomizationMode,
    ResponseTransformation,
    ResponseTransformationOption,
    WandbProject,
    resolve_response_transformation,
)
from ._quality import MinAbsFoldChange, MinRelevanceScore, NoQualityFilter, quality_from_cli
from ._summary import format_grouped_summary, resolve_group_by_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from drevalpy.models import ModelSpec
    from drevalpy.types.results.failure import RunFailure


def _model_specs(models: list[str], sample: list[str], drug: list[str], predictors: list[str]) -> list[ModelSpec]:
    """Build the model specs of the MODEL arguments and of the ``-S/-D/-P`` grid, one per name.

    :param models: Zoo names or recipes given as MODEL arguments.
    :param sample: Sample featurizer recipes of the grid.
    :param drug: Drug featurizer recipes of the grid.
    :param predictors: Predictors of the grid.
    :returns: The specs in the order given, without a second spec of the same name.
    :raises typer.BadParameter: If neither a model nor a predictor is given, or featurizers are given without a
        predictor.
    """
    from drevalpy.models import ModelSpec, combinations

    if not models and not predictors:
        raise typer.BadParameter("Give at least one MODEL or one --predictor/-P.")
    if (sample or drug) and not predictors:
        raise typer.BadParameter("--sample-featurizer/-S and --drug-featurizer/-D need at least one --predictor/-P.")
    specs = [ModelSpec.from_name(m) for m in models]
    if predictors:
        grid = combinations(sample=sample, drug=drug, predictors=predictors)
        for skipped in grid.skipped:
            typer.echo(f"Skipped {skipped.recipe}: {skipped.reason}")
        specs.extend(grid.specs)
    unique: dict[str, ModelSpec] = {}
    for spec in specs:
        unique.setdefault(spec.name, spec)
    return list(unique.values())


def _echo_failures(failures: Sequence[RunFailure]) -> None:
    """Print the failed runs of an experiment to stderr.

    :param failures: The recorded failures.
    """
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
    hpo: Hpo = True,
    hpo_metric: HpoMetric = Metric.RMSE,
    hpo_num_trials: HpoNumTrials = 16,
    seed: Annotated[
        int, typer.Option("--seed", help="Root seed for splits, HPO, feature randomization, and model fits.")
    ] = 42,
    hpo_random_state: HpoRandomState = None,
    randomization_mode: Annotated[
        list[RandomizationMode] | None,
        typer.Option(
            "--randomization-mode", case_sensitive=False, help="Randomization mode(s): SVRC, SVCC, SVRD, SVCD."
        ),
    ] = None,
    randomization_type: Annotated[
        RandomizationType, typer.Option("--randomization-type", case_sensitive=False, help="How to replace the view.")
    ] = RandomizationType.PERMUTATION,
    randomization_random_state: Annotated[
        int | None,
        typer.Option("--randomization-random-state", help="Root seed for feature randomization (default: --seed)."),
    ] = None,
    robustness_repeats: Annotated[
        int, typer.Option("--robustness-repeats", help="Robustness repeats per fold (0 = none).")
    ] = 0,
    response_transformation: ResponseTransformationOption = ResponseTransformation.NONE,
    group_by: GroupByOption = None,
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
    wandb_project: WandbProject = None,
    n_jobs: NJobs = 1,
) -> None:
    """Run the full evaluation pipeline."""
    from drevalpy._run import run

    transformer = resolve_response_transformation(response_transformation)
    model_specs = _model_specs(models or [], sample_featurizer or [], drug_featurizer or [], predictor or [])
    group_dataset = resolve_group_by_dataset(group_by, dataset)
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = run(
        models=model_specs,
        dataset=dataset if group_dataset is None else group_dataset,
        split_mode=split_mode,
        randomization_modes=[mode.value for mode in randomization_mode] if randomization_mode else None,
        randomization_type=randomization_type,
        seed=seed,
        randomization_random_state=randomization_random_state,
        hyperparameter_tuning=hpo,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        robustness_repeats=robustness_repeats,
        response_transformation=transformer,
        quality=quality_from_cli(no_quality_filter, min_relevance_score, min_abs_fold_change),
        wandb_project=wandb_project,
        n_jobs=n_jobs,
    )

    result.save(str(out))
    typer.echo(f"Wrote experiment results to {out}")
    typer.echo(repr(result))
    if group_by is not None:
        typer.echo(format_grouped_summary(result, group_by, group_dataset))
    if result.failures:
        _echo_failures(result.failures)
        raise typer.Exit(code=1)
