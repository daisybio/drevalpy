"""Typer option definitions shared by several commands, and the values they resolve to."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import typer

from drevalpy.evaluation import Metric

from ._summary import normalize_group_by

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer


class ResponseTransformation(StrEnum):
    """Values of ``--response-transformation``; ``none`` maps to no transformer."""

    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    DRUG_MEAN = "drug_mean"
    DRUG_TISSUE_MEAN = "drug_tissue_mean"


class RandomizationMode(StrEnum):
    """Randomization modes of ``run --randomization-mode`` and ``experiments randomization --mode``."""

    SVRC = "SVRC"
    SVCC = "SVCC"
    SVRD = "SVRD"
    SVCD = "SVCD"


ResponseTransformationOption = Annotated[
    ResponseTransformation,
    typer.Option(
        "--response-transformation",
        case_sensitive=False,
        help="Response transformation fitted on the training scope.",
    ),
]
Hpo = Annotated[bool, typer.Option("--hpo/--no-hpo", help="Enable hyperparameter tuning.")]
HpoMetric = Annotated[Metric, typer.Option("--hpo-metric", help="Metric to optimize.")]
HpoNumTrials = Annotated[int, typer.Option("--hpo-trials", help="Number of HPO trials.")]
HpoRandomState = Annotated[int | None, typer.Option("--hpo-random-state", help="HPO sampler seed (default: --seed).")]
WandbProject = Annotated[
    str | None,
    typer.Option("--wandb-project", help="Weights & Biases project to log the trials and the final fit to."),
]
NJobs = Annotated[
    int,
    typer.Option(
        "--n-jobs",
        help="Worker processes a single-drug model fits its per-drug models in; -1 uses all CPUs. "
        "Other models ignore it. The results do not depend on it.",
    ),
]
GroupByOption = Annotated[
    str | None,
    typer.Option(
        "--group-by",
        callback=normalize_group_by,
        help="Also print the metrics computed within each group instead of pooled over all pairs: drug, sample, "
        "study, or a sample/drug annotation column of the dataset such as tissue or ncit_class "
        "(sample:<column> or drug:<column> if a name is on both).",
    ),
]


def resolve_response_transformation(value: ResponseTransformation) -> ResponseTransformer | None:
    """Turn the ``--response-transformation`` choice into the response transformation prototype the library takes.

    :param value: The option value.
    :returns: An unfitted response transformation, or ``None`` for ``none``.
    """
    from drevalpy.models.response_transformation import get_response_transformation

    return get_response_transformation(value.value)
