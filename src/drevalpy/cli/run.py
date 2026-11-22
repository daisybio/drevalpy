"""``drevalpy run`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from drevalpy.evaluation import Metric
from drevalpy.models.response_transformation import ResponseTransformation, get_response_transformation
from drevalpy.types.enums import RandomizationMode, RandomizationType

from ._options import (
    DrugFeaturizerOption,
    GroupByOption,
    Hpo,
    HpoMetric,
    HpoNumTrials,
    HpoRandomState,
    Measure,
    MinAbsFoldChange,
    MinRelevanceScore,
    ModelsArgument,
    NJobs,
    NoQualityFilter,
    NSplits,
    PredictorOption,
    ResponseTransformationOption,
    SampleFeaturizerOption,
    ValidationRatio,
    WandbProject,
    quality_from_cli,
    resolve_model_specs,
)
from ._summary import format_grouped_summary, resolve_group_by_dataset


def run_cmd(
    dataset: Annotated[
        list[str],
        typer.Option("--dataset", "-d", help="Dataset name or .h5mu path; repeat to concatenate several studies."),
    ],
    models: ModelsArgument = None,
    split_mode: Annotated[
        str, typer.Option("--split-mode", "-s", help="Split mode: LPO, LSO, LDO, LTO, LStO.")
    ] = "LPO",
    n_splits: NSplits = 5,
    validation_ratio: ValidationRatio = 0.1,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for results. Each run is saved when it finishes, and rerunning the same command "
            "resumes a killed run.",
        ),
    ] = "results",
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Delete the drevalpy results in --output-dir first; needed when a run with other settings wrote "
            "them. Other files in the directory are kept.",
        ),
    ] = False,
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
    sample_featurizer: SampleFeaturizerOption = None,
    drug_featurizer: DrugFeaturizerOption = None,
    predictor: PredictorOption = None,
    measure: Measure = None,
    no_quality_filter: NoQualityFilter = False,
    min_relevance_score: MinRelevanceScore = None,
    min_abs_fold_change: MinAbsFoldChange = None,
    wandb_project: WandbProject = None,
    n_jobs: NJobs = 1,
) -> None:
    """Run the full evaluation pipeline."""
    from drevalpy._run import run
    from drevalpy.types.results.failure import ExperimentFailedError

    transformer = get_response_transformation(response_transformation.value)
    model_specs = resolve_model_specs(models or [], sample_featurizer or [], drug_featurizer or [], predictor or [])
    group_dataset = resolve_group_by_dataset(group_by, dataset)

    result = run(
        models=model_specs,
        dataset=dataset if group_dataset is None else group_dataset,
        split_mode=split_mode,
        n_splits=n_splits,
        validation_ratio=validation_ratio,
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
        measure=measure,
        wandb_project=wandb_project,
        n_jobs=n_jobs,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    typer.echo(f"Wrote experiment results to {UPath(output_dir)}")
    typer.echo(repr(result))
    if group_by is not None:
        typer.echo(format_grouped_summary(result, group_by, group_dataset))
    if result.failures:
        raise ExperimentFailedError(result.failures)
