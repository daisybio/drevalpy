"""The curve-quality options of ``run`` and ``data split``, and the rule they describe."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality

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
