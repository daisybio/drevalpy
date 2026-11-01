"""Typer option definitions shared by several commands, and the values they resolve to.

The groups: hyperparameter tuning (``--hpo``, ``--hpo-metric``, ``--hpo-trials``, ``--hpo-random-state``,
``--wandb-project``), ``--response-transformation``, ``--n-jobs``, ``--group-by`` and the curve-quality rule
(``--no-quality-filter``, ``--min-relevance-score``, ``--min-abs-fold-change``).
"""

from __future__ import annotations

from dataclasses import replace
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import typer

from drevalpy.evaluation import Metric

from ._summary import normalize_group_by

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality
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

NoQualityFilter = Annotated[
    bool,
    typer.Option("--no-quality-filter", help="Keep every pair with a finite response; apply no curve-quality rule."),
]
MinRelevanceScore = Annotated[
    float | None,
    typer.Option("--min-relevance-score", help="Minimum relevance score of a curve (default: -log10(0.05))."),
]
MinAbsFoldChange = Annotated[
    float | None,
    typer.Option("--min-abs-fold-change", help="Minimum absolute log2 fold change of a curve (default: 0.45)."),
]


def resolve_response_transformation(value: ResponseTransformation) -> ResponseTransformer | None:
    """Turn the ``--response-transformation`` choice into the response transformation prototype the library takes.

    :param value: The option value.
    :returns: An unfitted response transformation, or ``None`` for ``none``.
    """
    from drevalpy.models.response_transformation import get_response_transformation

    return get_response_transformation(value.value)


def quality_from_cli(
    no_filter: bool,
    min_relevance_score: float | None,
    min_abs_fold_change: float | None,
) -> CurveQuality:
    """Build the curve-quality rule the command-line options ask for.

    :param no_filter: ``--no-quality-filter``: no rule at all.
    :param min_relevance_score: ``--min-relevance-score``, or ``None`` for the default threshold.
    :param min_abs_fold_change: ``--min-abs-fold-change``, or ``None`` for the default threshold.
    :returns: ``CurveQuality.off()``, or the default rule with every given threshold replaced.
    :raises typer.BadParameter: If *no_filter* is combined with a threshold.
    """
    from drevalpy.data.quality import CurveQuality

    overrides = {
        "min_relevance_score": min_relevance_score,
        "min_abs_fold_change": min_abs_fold_change,
    }
    given = {name: value for name, value in overrides.items() if value is not None}
    if no_filter and given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise typer.BadParameter(f"{flags} cannot be combined with --no-quality-filter.")
    if no_filter:
        return CurveQuality.off()
    return replace(CurveQuality.default(), **given)
