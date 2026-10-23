"""``drevalpy train`` command."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from upath import UPath

from ._options import (
    Hpo,
    HpoMetric,
    HpoNumTrials,
    HpoRandomState,
    NJobs,
    ResponseTransformation,
    ResponseTransformationOption,
    resolve_response_transformation,
)
from ._quality import MinAbsFoldChange, MinRelevanceScore, NoQualityFilter, quality_from_cli

#: File of the model directory that holds the checkpoint.
MODEL_FILE = "model.zip"
#: File of the model directory that records how the model was trained.
METADATA_FILE = "metadata.json"


def train_cmd(
    model: Annotated[str, typer.Argument(help="Zoo name or recipe, such as ElasticNet or pca[gene_expression]:ridge.")],
    dataset: Annotated[
        str, typer.Option("--dataset", "-d", help="Dataset name or .h5mu path; every observed pair is used.")
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Model directory to create; it receives model.zip and metadata.json.")
    ],
    hpo: Hpo = True,
    hpo_metric: HpoMetric = "RMSE",
    hpo_num_trials: HpoNumTrials = 16,
    seed: Annotated[
        int, typer.Option("--seed", help="Root seed for the validation slice, HPO and the model fit.")
    ] = 42,
    hpo_random_state: HpoRandomState = None,
    response_transformation: ResponseTransformationOption = ResponseTransformation.NONE,
    no_quality_filter: NoQualityFilter = False,
    min_relevance_score: MinRelevanceScore = None,
    min_abs_fold_change: MinAbsFoldChange = None,
    n_jobs: NJobs = 1,
) -> None:
    """Train a final model on all pairs of a dataset and save it."""
    from drevalpy import __version__
    from drevalpy._train import train
    from drevalpy.data import load
    from drevalpy.models import ModelSpec

    spec = ModelSpec.from_name(model)
    transformer = resolve_response_transformation(response_transformation)
    quality = quality_from_cli(no_quality_filter, min_relevance_score, min_abs_fold_change)
    data = load(dataset)
    fitted = train(
        spec,
        data,
        hyperparameter_tuning=hpo,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        response_transformation=transformer,
        quality=quality,
        seed=seed,
        n_jobs=n_jobs,
    )

    out = UPath(output)
    out.mkdir(parents=True, exist_ok=True)
    fitted.save(out / MODEL_FILE)
    metadata = {
        "model_name": fitted.name,
        "dataset_name": data.name,
        "seed": seed,
        "response_transformation": response_transformation.value,
        "hyperparameter_tuning": hpo,
        "drevalpy_version": __version__,
    }
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    typer.echo(f"Model: {fitted.name} trained on {data.name} -> {out}")
