"""Typer option definitions shared by several commands, and the values they resolve to."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import typer

from ._summary import GroupBy

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


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
HpoMetric = Annotated[str, typer.Option("--hpo-metric", help="Metric to optimize.")]
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
    GroupBy | None,
    typer.Option(
        "--group-by",
        case_sensitive=False,
        help="Also print the metrics computed within each drug, sample or study instead of pooled over all pairs.",
    ),
]


def resolve_response_transformation(value: ResponseTransformation) -> TransformerMixin | None:
    """Turn the ``--response-transformation`` choice into the transformer prototype the library takes.

    :param value: The option value.
    :returns: An unfitted sklearn transformer, or ``None`` for ``none``.
    """
    from drevalpy.utils import get_response_transformation

    return get_response_transformation(value.value)
